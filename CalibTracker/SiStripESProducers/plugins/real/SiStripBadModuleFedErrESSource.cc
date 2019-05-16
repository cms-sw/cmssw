// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripBadModuleFedErrESSource
//
/**\class SiStripBadModuleFedErrESSource SiStripBadModuleFedErrESSource.h CalibTracker/SiStripESProducers/plugins/SiStripBadModuleFedErrESSource.cc

 Description: SiStripBadStrip ESProducer from FED errors

  A new SiStrip fake source has been added to create Bad Components from the list of Fed detected errors. This
  is done using a histogram from DQM output where FedId vs APVId is plotted for detected channels.

 Implementation:

  - accesses the specific histogram from the DQM root file (to be specified in the configuration) and creates
   `SiStripBadStrip` object checking detected `FedChannel` and `FedId` and using `SiStripFedCabling` information.
  - the record `SiStripBadModuleFedErrRcd` is defined in `CalibTracker/Records` package in `SiStripDependentRecords.h`
    and `SiStripDependentRecords.cc`. This is a dependent record and depends on `SiStripFedCablingRcd'. This record
    is filled with `SiStripBadStrip` object.
  - corresponding configuration file is `CalibTracker/SiStripESProducers/python/fake/SiStripBadModuleFedErrESSource_cfi.py`
  - An overall configuration file can be found in `CalibTracker/SiStripESProducers/test/mergeBadChannel_cfg.py` which merges
    SiStrip Bad channels from PLC, `RunInfo` and SiStripBadModuleFedErrESSource in `SiStripQualityEsProducer` and finally
    listed by the `SiStripQualityStatistics` module.

 Original author: Suchandra Dutta <Suchandra.Dutta@cern.ch>

 Port to edm::ESProducer: Pieter David <Pieter.David@cern.ch>, summer 2016
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"


class SiStripBadModuleFedErrESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripBadModuleFedErrESSource(const edm::ParameterSet&);
  ~SiStripBadModuleFedErrESSource() override;

  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity ) override;

  typedef std::unique_ptr<SiStripBadStrip> ReturnType;
  ReturnType produce( const SiStripBadModuleFedErrRcd& );

private:
  bool m_readFlag;
  std::string m_fileName;
  float m_cutoff;

  std::vector<std::pair<uint16_t,uint16_t>> getFedBadChannelList( DQMStore* dqmStore, const MonitorElement* me ) const;
  float getProcessedEvents( DQMStore* dqmStore ) const;
};

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DQMServices/Core/interface/DQMStore.h"

SiStripBadModuleFedErrESSource::SiStripBadModuleFedErrESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripBadModuleFedErrRcd>();

  m_readFlag = iConfig.getParameter<bool>("ReadFromFile");
  m_fileName = iConfig.getParameter<std::string>("FileName");
  m_cutoff = iConfig.getParameter<double>("BadStripCutoff");
}

SiStripBadModuleFedErrESSource::~SiStripBadModuleFedErrESSource() {}

void SiStripBadModuleFedErrESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity )
{
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

float SiStripBadModuleFedErrESSource::getProcessedEvents( DQMStore* dqmStore ) const
{
  dqmStore->cd();

  const std::string dname{"SiStrip/ReadoutView"};
  const std::string hpath{dname+"/nTotalBadActiveChannels"};
  if ( dqmStore->dirExists(dname) ) {
    MonitorElement* me = dqmStore->get(hpath);
    if (me) return me->getEntries();
  }
  return 0;
}

std::vector<std::pair<uint16_t,uint16_t>> SiStripBadModuleFedErrESSource::getFedBadChannelList( DQMStore* dqmStore, const MonitorElement* me ) const
{
  std::vector<std::pair<uint16_t,uint16_t>> ret;
  if (me->kind() == MonitorElement::DQM_KIND_TH2F) {
    TH2F* th2 = me->getTH2F();
    float entries = getProcessedEvents(dqmStore);
    if ( ! entries ) entries = th2->GetBinContent(th2->GetMaximumBin());
    for ( uint16_t i = 1; i < th2->GetNbinsY()+1; ++i ) {
      for ( uint16_t j = 1; j < th2->GetNbinsX()+1; ++j ) {
        if ( th2->GetBinContent(j,i) > m_cutoff * entries ) {
          edm::LogInfo("SiStripBadModuleFedErrService") << " [SiStripBadModuleFedErrService::getFedBadChannelList] :: FedId & Channel " << th2->GetYaxis()->GetBinLowEdge(i) <<   "  " << th2->GetXaxis()->GetBinLowEdge(j);
          ret.push_back(std::pair<uint16_t, uint16_t>(th2->GetYaxis()->GetBinLowEdge(i), th2->GetXaxis()->GetBinLowEdge(j)));
        }
      }
    }
  }
  return ret;
}

// ------------ method called to produce the data  ------------
SiStripBadModuleFedErrESSource::ReturnType
SiStripBadModuleFedErrESSource::produce(const SiStripBadModuleFedErrRcd& iRecord)
{
  using namespace edm::es;

  edm::ESHandle<SiStripFedCabling> cabling;
  iRecord.getRecord<SiStripFedCablingRcd>().get(cabling);

  auto quality = std::make_unique<SiStripQuality>();

  DQMStore* dqmStore = edm::Service<DQMStore>().operator->();
  if ( m_readFlag ) { // open requested file
    edm::LogInfo("SiStripBadModuleFedErrService") <<  "[SiStripBadModuleFedErrService::openRequestedFile] Accessing root File" << m_fileName;
    if ( ! dqmStore->load(m_fileName, DQMStore::OpenRunDirs::StripRunDirs, true) ) {
      edm::LogError("SiStripBadModuleFedErrService")<<"[SiStripBadModuleFedErrService::openRequestedFile] Requested file " << m_fileName << "Can not be opened!! ";
      return quality;
    }
  }

  dqmStore->cd();

  const std::string dname{"SiStrip/ReadoutView"};
  const std::string hpath{dname + "/FedIdVsApvId"};
  if ( dqmStore->dirExists(dname) ) {
    MonitorElement* me = dqmStore->get(hpath);
    if ( me ) {
      std::map<uint32_t, std::set<int>> detectorMap;
      for ( const auto& elm : getFedBadChannelList(dqmStore, me) ) {
	const uint16_t fId = elm.first;
	const uint16_t fChan = elm.second/2;
        if ( ( fId == 9999 ) && ( fChan == 9999 ) ) continue;

	FedChannelConnection channel = cabling->fedConnection(fId, fChan);
        detectorMap[channel.detId()].insert(channel.apvPairNumber());
      }

      for ( const auto& detElm : detectorMap ) { // pair(detId, pairs)
        SiStripQuality::InputVector theSiStripVector;
	unsigned short firstBadStrip{0};
	unsigned short fNconsecutiveBadStrips{0};
        int last_pair = -1;
        for ( const auto pair : detElm.second ) {
          if ( last_pair == -1 ) {
	    firstBadStrip = pair * 128*2;
	    fNconsecutiveBadStrips = 128*2;
	  } else if ( pair - last_pair  > 1 ) {
	    theSiStripVector.push_back(quality->encode(firstBadStrip, fNconsecutiveBadStrips));
	    firstBadStrip = pair * 128*2;
	    fNconsecutiveBadStrips = 128*2;
	  } else {
	    fNconsecutiveBadStrips += 128*2;
	  }
          last_pair = pair;
        }
        unsigned int theBadStripRange = quality->encode(firstBadStrip, fNconsecutiveBadStrips);
	theSiStripVector.push_back(theBadStripRange);

	edm::LogInfo("SiStripBadModuleFedErrService")
            << " SiStripBadModuleFedErrService::readBadComponentsFromFed "
            << " detid " << detElm.first
            << " firstBadStrip " << firstBadStrip
            << " NconsecutiveBadStrips " << fNconsecutiveBadStrips
            << " packed integer " << std::hex << theBadStripRange << std::dec;

	if ( ! quality->put(detElm.first, SiStripBadStrip::Range{theSiStripVector.begin(), theSiStripVector.end()}) ) {
	  edm::LogError("SiStripBadModuleFedErrService") << "[SiStripBadModuleFedErrService::readBadComponentsFromFed] detid already exists";
	}
      }
      quality->cleanUp();
    }
  }

  return quality;
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBadModuleFedErrESSource);
