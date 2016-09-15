// -*- C++ -*-
//
// Package:    CalibTracker/SiStripESProducers
// Class:      SiStripBadModuleConfigurableFakeESSource
//
/**\class SiStripBadModuleConfigurableFakeESSource SiStripBadModuleConfigurableFakeESSource.h CalibTracker/SiStripESProducers/plugins/SiStripBadModuleConfigurableFakeESSource.cc

 Description: "fake" SiStripBadStrip ESProducer - configurable list of bad modules

 Implementation:
     Port of SiStripBadModuleGenerator and templated fake ESSource to an edm::ESProducer
*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripBadModuleConfigurableFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripBadModuleConfigurableFakeESSource(const edm::ParameterSet&);
  ~SiStripBadModuleConfigurableFakeESSource();

  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity );

  typedef std::shared_ptr<SiStripBadStrip> ReturnType;
  ReturnType produce(const SiStripBadModuleRcd&);

private:
  using Parameters = std::vector<edm::ParameterSet>;
  Parameters m_badComponentList;
  edm::FileInPath m_file;
  bool m_printDebug;

  void selectDetectors(const TrackerTopology* tTopo, const std::vector<uint32_t>& detIds, std::vector<uint32_t>& selList) const;
};

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

SiStripBadModuleConfigurableFakeESSource::SiStripBadModuleConfigurableFakeESSource(const edm::ParameterSet& iConfig)
{
  setWhatProduced(this);
  findingRecord<SiStripBadModuleRcd>();

  m_badComponentList = iConfig.getUntrackedParameter<Parameters>("BadComponentList");
  m_file = iConfig.getParameter<edm::FileInPath>("file");
  m_printDebug = iConfig.getUntrackedParameter<bool>("printDebug", false);
}

SiStripBadModuleConfigurableFakeESSource::~SiStripBadModuleConfigurableFakeESSource() {}

void SiStripBadModuleConfigurableFakeESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue& iov, edm::ValidityInterval& iValidity )
{
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

// ------------ method called to produce the data  ------------
SiStripBadModuleConfigurableFakeESSource::ReturnType
SiStripBadModuleConfigurableFakeESSource::produce(const SiStripBadModuleRcd& iRecord)
{
  using namespace edm::es;

  edm::ESHandle<TrackerTopology> tTopo;
  iRecord.getRecord<TrackerTopologyRcd>().get(tTopo);

  std::shared_ptr<SiStripQuality> quality{new SiStripQuality};

  SiStripDetInfoFileReader reader{m_file.fullPath()};

  std::vector<uint32_t> selDetIds;
  selectDetectors(tTopo.product(), reader.getAllDetIds(), selDetIds);
  edm::LogInfo("SiStripQualityConfigurableFakeESSource")<<"[produce] number of selected dets to be removed " << selDetIds.size() <<std::endl;

  std::stringstream ss;
  for ( const auto selId : selDetIds ) {
    SiStripQuality::InputVector theSiStripVector;

    unsigned short firstBadStrip{0};
    unsigned short NconsecutiveBadStrips = reader.getNumberOfApvsAndStripLength(selId).first * 128;
    unsigned int theBadStripRange{quality->encode(firstBadStrip,NconsecutiveBadStrips)};

    if (m_printDebug) {
      ss << "detid " << selId << " \t"
	 << " firstBadStrip " << firstBadStrip << "\t "
	 << " NconsecutiveBadStrips " << NconsecutiveBadStrips << "\t "
	 << " packed integer " << std::hex << theBadStripRange  << std::dec
	 << std::endl;
    }

    theSiStripVector.push_back(theBadStripRange);

    if ( ! quality->put(selId,SiStripBadStrip::Range{theSiStripVector.begin(),theSiStripVector.end()}) ) {
      edm::LogError("SiStripQualityConfigurableFakeESSource") << "[produce] detid already exists";
    }
  }
  if (m_printDebug) {
    edm::LogInfo("SiStripQualityConfigurableFakeESSource") << ss.str();
  }
  quality->cleanUp();
  //quality->fillBadComponents();

  if (m_printDebug){
    std::stringstream ss1;
    for ( const auto& badComp : quality->getBadComponentList() ) {
      ss1 << "bad module " << badComp.detid << " " << badComp.BadModule <<  "\n";
    }
    edm::LogInfo("SiStripQualityConfigurableFakeESSource") << ss1.str();
  }

  return quality;
}

namespace {
  bool isTIBDetector(const TrackerTopology* tTopo,
                     const DetId & therawid,
		     uint32_t requested_layer,
		     uint32_t requested_bkw_frw,
		     uint32_t requested_int_ext,
		     uint32_t requested_string,
		     uint32_t requested_ster,
		     uint32_t requested_detid)
  {
    // check if subdetector field is a TIB, both tested numbers are int
    return ( ( therawid.subdetId() == SiStripDetId::TIB )
           // check if TIB is from the ones requested
           // take everything if default value is 0
          && ( ( tTopo->tibLayer(therawid) == requested_layer )
            || ( requested_layer == 0 ) )
          && ( ( tTopo->tibIsZPlusSide(therawid) && ( requested_bkw_frw == 2 ) )
            || ( ( ! tTopo->tibIsZPlusSide(therawid) ) && ( requested_bkw_frw == 1 ) )
            || ( requested_bkw_frw == 0 ) )
          && ( ( tTopo->tibIsInternalString(therawid) && ( requested_int_ext == 1 ) )
            || ( ( ! tTopo->tibIsInternalString(therawid) ) && ( requested_int_ext == 2 ) )
            || ( requested_int_ext == 0 ) )
          && ( ( tTopo->tibIsStereo(therawid) && ( requested_ster == 1 ) )
            || ( tTopo->tibIsRPhi(therawid) && ( requested_ster == 2 ) )
            || ( requested_ster == 0 ) )
          && ( ( tTopo->tibString(therawid) == requested_string )
            || ( requested_string == 0 ) )
          && ( ( therawid.rawId() == requested_detid )
            || ( requested_detid == 0 ) )
          );
  }

  bool isTOBDetector(const TrackerTopology* tTopo,
                     const DetId & therawid,
		     uint32_t requested_layer,
		     uint32_t requested_bkw_frw,
		     uint32_t requested_rod,
		     uint32_t requested_ster,
		     uint32_t requested_detid)
  {
    // check if subdetector field is a TOB, both tested numbers are int
    return ( ( therawid.subdetId() ==  SiStripDetId::TOB )
           // check if TOB is from the ones requested
           // take everything if default value is 0
          && ( ( tTopo->tobLayer(therawid) == requested_layer )
            || ( requested_layer == 0 ) )
          && ( ( tTopo->tobIsZPlusSide(therawid) && ( requested_bkw_frw == 2 ) )
            || ( ! tTopo->tobIsZPlusSide(therawid) && ( requested_bkw_frw == 1 ) )
            || ( requested_bkw_frw == 0 ) )
          && ( ( tTopo->tobIsStereo(therawid) && ( requested_ster == 1 ) )
            || ( tTopo->tobIsRPhi(therawid) && ( requested_ster == 2 ) )
            || ( requested_ster == 0 ) )
          && ( ( tTopo->tobRod(therawid) == requested_rod )
            || ( requested_rod == 0 ) )
          && ( ( therawid.rawId() == requested_detid )
            || ( requested_detid == 0 ) )
          );
  }

  bool isTIDDetector(const TrackerTopology* tTopo,
                     const DetId & therawid,
		     uint32_t requested_side,
		     uint32_t requested_wheel,
		     uint32_t requested_ring,
		     uint32_t requested_ster,
		     uint32_t requested_detid)
  {
    // check if subdetector field is a TID, both tested numbers are int
    return ( ( therawid.subdetId() ==  SiStripDetId::TID )
           // check if TID is from the ones requested
           // take everything if default value is 0
          && ( ( tTopo->tidWheel(therawid) == requested_wheel )
            || ( requested_wheel == 0 ) )
          && ( ( tTopo->tidIsZPlusSide(therawid) && ( requested_side == 2 ) )
            || ( ! tTopo->tidIsZPlusSide(therawid) && ( requested_side == 1 ) )
            || ( requested_side == 0 ) )
          && ( ( tTopo->tidIsStereo(therawid) && ( requested_ster == 1 ) )
            || ( tTopo->tidIsRPhi(therawid) && ( requested_ster == 2 ) )
            || ( requested_ster == 0 ) )
          && ( ( tTopo->tidRing(therawid) == requested_ring )
            || ( requested_ring == 0 ) )
          && ( ( therawid.rawId() == requested_detid )
            || ( requested_detid == 0 ) )
          );
  }

  bool isTECDetector(const TrackerTopology* tTopo,
                     const DetId & therawid,
		     uint32_t requested_side,
		     uint32_t requested_wheel,
		     uint32_t requested_petal_bkw_frw,
		     uint32_t requested_petal,
		     uint32_t requested_ring,
		     uint32_t requested_ster,
		     uint32_t requested_detid)
  {
    // check if subdetector field is a TEC, both tested numbers are int
    return ( ( therawid.subdetId()  ==   SiStripDetId::TEC )
           // check if TEC is from the ones requested 
           // take everything if default value is 0
          && ( ( tTopo->tecWheel(therawid) == requested_wheel )
            || ( requested_wheel == 0 ) )
          && ( ( tTopo->tecIsZPlusSide(therawid) && ( requested_side == 2 ) )
            || ( ( ! tTopo->tecIsZPlusSide(therawid) ) && ( requested_side == 1 ) )
            || ( requested_side == 0 ) )
          && ( ( tTopo->tecIsStereo(therawid) && ( requested_ster == 1 ) )
            || ( ( ! tTopo->tecIsStereo(therawid) ) && ( requested_ster == 2 ) )
            || ( requested_ster == 0 ) )
          && ( ( tTopo->tecIsFrontPetal(therawid) && ( requested_petal_bkw_frw == 2 ) )
            || ( ( ! tTopo->tecIsFrontPetal(therawid) ) && ( requested_petal_bkw_frw == 2 ) )
            || ( requested_petal_bkw_frw == 0 ) )
          && ( ( tTopo->tecPetalNumber(therawid) == requested_petal )
            || ( requested_petal == 0 ) )
          && ( ( tTopo->tecRing(therawid) == requested_ring )
            || ( requested_ring == 0 ) )
          && ( ( therawid.rawId() == requested_detid )
            || ( requested_detid == 0 ) )
          );
  }
}

void SiStripBadModuleConfigurableFakeESSource::selectDetectors(const TrackerTopology* tTopo, const std::vector<uint32_t>& detIds, std::vector<uint32_t>& selList) const
{
  std::stringstream ss;
  for ( const auto& badComp : m_badComponentList ) {
    const std::string subDetStr{badComp.getParameter<std::string>("SubDet")};
    if (m_printDebug) ss << "Bad SubDet " << subDetStr << " \t";

    SiStripDetId::SubDetector    subDet = SiStripDetId::UNKNOWN;
    if      ( subDetStr=="TIB" ) subDet = SiStripDetId::TIB;
    else if ( subDetStr=="TID" ) subDet = SiStripDetId::TID;
    else if ( subDetStr=="TOB" ) subDet = SiStripDetId::TOB;
    else if ( subDetStr=="TEC" ) subDet = SiStripDetId::TEC;

    std::vector<uint32_t> genericBadDetIds(badComp.getUntrackedParameter<std::vector<uint32_t> >("detidList", std::vector<uint32_t>()));
    bool anySubDet{ ! genericBadDetIds.empty() };

    std::cout << "genericBadDetIds.size() = " << genericBadDetIds.size() << std::endl;

    uint32_t startDet{DetId{DetId::Tracker, anySubDet ? SiStripDetId::TIB   : subDet  }.rawId()};
    uint32_t stopDet {DetId{DetId::Tracker, anySubDet ? SiStripDetId::TEC+1 : subDet+1}.rawId()};

    for ( std::vector<uint32_t>::const_iterator iter{lower_bound(detIds.begin(), detIds.end(), startDet)}
        , iterEnd{lower_bound(detIds.begin(), detIds.end(), stopDet)}
        ; iter != iterEnd; ++iter )
    {
      const DetId detectorId{*iter};
      bool resp{false};
      switch (subDet) {
        case SiStripDetId::TIB:
          resp = isTIBDetector(tTopo,detectorId,
                               badComp.getParameter<uint32_t>("layer"),
                               badComp.getParameter<uint32_t>("bkw_frw"),
                               badComp.getParameter<uint32_t>("int_ext"),
                               badComp.getParameter<uint32_t>("ster"),
                               badComp.getParameter<uint32_t>("string_"),
                               badComp.getParameter<uint32_t>("detid")
                               );
          break;
        case SiStripDetId::TID:
          resp = isTIDDetector(tTopo,detectorId,
                               badComp.getParameter<uint32_t>("side"),
                               badComp.getParameter<uint32_t>("wheel"),
                               badComp.getParameter<uint32_t>("ring"),
                               badComp.getParameter<uint32_t>("ster"),
                               badComp.getParameter<uint32_t>("detid")
                               );
          break;
        case SiStripDetId::TOB:
          resp = isTOBDetector(tTopo,detectorId,
                               badComp.getParameter<uint32_t>("layer"),
                               badComp.getParameter<uint32_t>("bkw_frw"),
                               badComp.getParameter<uint32_t>("rod"),
                               badComp.getParameter<uint32_t>("ster"),
                               badComp.getParameter<uint32_t>("detid")
                               );
          break;
        case SiStripDetId::TEC:
          resp = isTECDetector(tTopo,detectorId,
                               badComp.getParameter<uint32_t>("side"),
                               badComp.getParameter<uint32_t>("wheel"),
                               badComp.getParameter<uint32_t>("petal_bkw_frw"),
                               badComp.getParameter<uint32_t>("petal"),
                               badComp.getParameter<uint32_t>("ring"),
                               badComp.getParameter<uint32_t>("ster"),
                               badComp.getParameter<uint32_t>("detid")
                               );
          break;
        default:
          break;
      }
      if ( anySubDet ) {
        std::cout << "AnySubDet" << *iter << std::endl;
        resp = ( std::find(genericBadDetIds.begin(), genericBadDetIds.end(), *iter) != genericBadDetIds.end() );
      }

      if ( resp ) {
        selList.push_back(*iter);
      }
    }
  }
  if (m_printDebug) {
    edm::LogInfo("SiStripBadModuleGenerator") << ss.str();
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripBadModuleConfigurableFakeESSource);
