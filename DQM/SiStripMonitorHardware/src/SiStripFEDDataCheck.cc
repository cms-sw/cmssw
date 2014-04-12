// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      SiStripFEDCheckPlugin
// 
/**\class SiStripFEDCheckPlugin SiStripFEDCheck.cc DQM/SiStripMonitorHardware/plugins/SiStripFEDCheck.cc

 Description: DQM source application to produce data integrety histograms for SiStrip data for use in HLT and Prompt reco
*/
//
// Original Author:  Nicholas Cripps
//         Created:  2008/09/16
//
//
#include <memory>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include "DQM/SiStripMonitorHardware/interface/FEDErrors.hh"


//
// Class declaration
//

class SiStripFEDCheckPlugin : public edm::EDAnalyzer
{
 public:
  explicit SiStripFEDCheckPlugin(const edm::ParameterSet&);
  ~SiStripFEDCheckPlugin();
 private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  virtual void endRun();

  bool hasFatalError(const FEDRawData& fedData, unsigned int fedId) const;
  bool hasNonFatalError(const FEDRawData& fedData, unsigned int fedId) const;
  void updateCabling(const edm::EventSetup& eventSetup);
  
  inline void fillPresent(unsigned int fedId, bool present);
  inline void fillFatalError(unsigned int fedId, bool fatalError);
  inline void fillNonFatalError(unsigned int fedId, float nonFatalError);
  
  void doUpdateIfNeeded();
  void updateHistograms();
  
  
  edm::InputTag rawDataTag_;
  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;
  std::string dirName_;
  bool printDebug_;
  bool writeDQMStore_;
  
  //Histograms
  DQMStore* dqm_;
  MonitorElement* fedsPresent_;
  MonitorElement* fedFatalErrors_;
  MonitorElement* fedNonFatalErrors_;
  
  //For histogram cache
  unsigned int updateFrequency_;//Update histograms with cached values every n events. If zero then fill normally every event
  //cache values
  std::vector<unsigned int> fedsPresentBinContents_;
  std::vector<unsigned int> fedFatalErrorBinContents_;
  std::vector<unsigned int> fedNonFatalErrorBinContents_;
  unsigned int eventCount_;//incremented by doUpdateIfNeeded()
  
  //Fine grained control of tests
  bool doPayloadChecks_, checkChannelLengths_, checkPacketCodes_, checkFELengths_, checkChannelStatusBits_;
  
  //Cabling
  uint32_t cablingCacheId_;
  const SiStripFedCabling* cabling_;
};


//
// Constructors and destructor
//

SiStripFEDCheckPlugin::SiStripFEDCheckPlugin(const edm::ParameterSet& iConfig)
  : rawDataTag_(iConfig.getParameter<edm::InputTag>("RawDataTag")),
    dirName_(iConfig.getUntrackedParameter<std::string>("DirName","SiStrip/FEDIntegrity/")),
    printDebug_(iConfig.getUntrackedParameter<bool>("PrintDebugMessages",false)),
    writeDQMStore_(iConfig.getUntrackedParameter<bool>("WriteDQMStore",false)),
    updateFrequency_(iConfig.getUntrackedParameter<unsigned int>("HistogramUpdateFrequency",0)),
    fedsPresentBinContents_(FEDNumbering::MAXSiStripFEDID+1,0),
    fedFatalErrorBinContents_(FEDNumbering::MAXSiStripFEDID+1,0),
    fedNonFatalErrorBinContents_(FEDNumbering::MAXSiStripFEDID+1,0),
    eventCount_(0),
    doPayloadChecks_(iConfig.getUntrackedParameter<bool>("DoPayloadChecks",true)),
    checkChannelLengths_(iConfig.getUntrackedParameter<bool>("CheckChannelLengths",true)),
    checkPacketCodes_(iConfig.getUntrackedParameter<bool>("CheckChannelPacketCodes",true)),
    checkFELengths_(iConfig.getUntrackedParameter<bool>("CheckFELengths",true)),
    checkChannelStatusBits_(iConfig.getUntrackedParameter<bool>("CheckChannelStatus",true)),
    cablingCacheId_(0)
{
  rawDataToken_ = consumes<FEDRawDataCollection>(rawDataTag_);
  if (printDebug_ && !doPayloadChecks_ && (checkChannelLengths_ || checkPacketCodes_ || checkFELengths_)) {
    std::stringstream ss;
    ss << "Payload checks are disabled but individual payload checks have been enabled. The following payload checks will be skipped: ";
    if (checkChannelLengths_) ss << "Channel length check, ";
    if (checkPacketCodes_) ss << "Channel packet code check, ";
    if (checkChannelStatusBits_) ss << "Cabled channel status bits checks, ";
    if (checkFELengths_) ss << "FE Unit legnth check";
    edm::LogWarning("SiStripFEDCheck") << ss.str();
  }
}

SiStripFEDCheckPlugin::~SiStripFEDCheckPlugin()
{
}


//
// Member functions
//

// ------------ method called to for each event  ------------
void
SiStripFEDCheckPlugin::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //update cabling
  updateCabling(iSetup);
  
  //get raw data
  edm::Handle<FEDRawDataCollection> rawDataCollectionHandle;
  const bool gotData = iEvent.getByToken(rawDataToken_,rawDataCollectionHandle);
  if (!gotData) {
    //module is required to silently do nothing when data is not present
    return;
  }
  const FEDRawDataCollection& rawDataCollection = *rawDataCollectionHandle;
  
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  
  //FED errors
  FEDErrors lFedErrors;

  //loop over siStrip FED IDs
  for (unsigned int fedId = siStripFedIdMin; fedId <= siStripFedIdMax; fedId++) {
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);

    //create an object to fill all errors
    //third param to false:save time by not initialising anything not used here
    lFedErrors.initialiseFED(fedId,cabling_,tTopo,false);


    //check data exists
    if (!fedData.size() || !fedData.data()) {
      fillPresent(fedId,0);
      continue;
    }
    //fill buffer present histogram
    fillPresent(fedId,1);

    //check for fatal errors
    //no need for debug output
    bool hasFatalErrors = false;
    float rateNonFatal = 0;

    std::auto_ptr<const sistrip::FEDBuffer> buffer;

    if (!lFedErrors.fillFatalFEDErrors(fedData,0)) {
      hasFatalErrors = true;
    }
    else {
      //need to construct full object to go any further
      if (doPayloadChecks_ || checkChannelStatusBits_) {
	
	buffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size(),true));
	if (doPayloadChecks_) {

	  bool channelLengthsOK = checkChannelLengths_ ? buffer->checkChannelLengthsMatchBufferLength() : true;
	  bool channelPacketCodesOK = checkPacketCodes_ ? buffer->checkChannelPacketCodes() : true;
	  bool feLengthsOK = checkFELengths_ ? buffer->checkFEUnitLengths() : true;
	  if ( !channelLengthsOK ||
	       !channelPacketCodesOK ||
	       !feLengthsOK ) {
	    hasFatalErrors = true;
	  }
	}
	if (checkChannelStatusBits_) rateNonFatal = lFedErrors.fillNonFatalFEDErrors(buffer.get(),cabling_);
      }
    }

    if (hasFatalErrors) {
      fillFatalError(fedId,1);
      if (printDebug_) {
	if (!buffer.get()) buffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size(),true));
	edm::LogInfo("SiStripFEDCheck") << "Fatal error with FED ID " << fedId << ". Check summary: " 
					<< std::endl << buffer->checkSummary() << std::endl;
	std::stringstream ss;
	buffer->dump(ss);
	edm::LogInfo("SiStripFEDCheck") << ss.str();
      }
    }
    else {
      fillFatalError(fedId,0);
      //fill non-fatal errors histogram if there were no fatal errors
      fillNonFatalError(fedId,rateNonFatal);
      if (printDebug_ && rateNonFatal > 0) {
	if (!buffer.get()) buffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size(),true));
	edm::LogInfo("SiStripFEDCheck") << "Non-fatal error with FED ID " << fedId 
					<< " for " << rateNonFatal << " of the channels. Check summary: " 
					<< std::endl << buffer->checkSummary() << std::endl;
	std::stringstream ss;
	buffer->dump(ss);
	edm::LogInfo("SiStripFEDCheck") << ss.str();
      }

    }
  }//loop over FED IDs
  
  //update histograms if needed
  doUpdateIfNeeded();
}

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripFEDCheckPlugin::beginJob()
{
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  //get DQM store
  dqm_ = &(*edm::Service<DQMStore>());
  dqm_->setCurrentFolder(dirName_);
  //book histograms
  fedsPresent_ = dqm_->book1D("FEDEntries",
                              "Number of times FED buffer is present in data",
                              siStripFedIdMax-siStripFedIdMin+1,
                              siStripFedIdMin-0.5,siStripFedIdMax+0.5);
  fedsPresent_->setAxisTitle("FED-ID",1);
  fedFatalErrors_ = dqm_->book1D("FEDFatal",
                              "Number of fatal errors in FED buffer",
                              siStripFedIdMax-siStripFedIdMin+1,
                              siStripFedIdMin-0.5,siStripFedIdMax+0.5);
  fedFatalErrors_->setAxisTitle("FED-ID",1);
  fedNonFatalErrors_ = dqm_->book1D("FEDNonFatal",
                              "Number of non fatal errors in FED buffer",
                              siStripFedIdMax-siStripFedIdMin+1,
                              siStripFedIdMin-0.5,siStripFedIdMax+0.5);
  fedNonFatalErrors_->setAxisTitle("FED-ID",1);
}

// ------------ method called once each run just after ending the event loop  ------------
void 
SiStripFEDCheckPlugin::endRun()
{
  updateHistograms();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripFEDCheckPlugin::endJob()
{
  if (writeDQMStore_) dqm_->save("DQMStore.root");
}


void SiStripFEDCheckPlugin::updateCabling(const edm::EventSetup& eventSetup)
{
  uint32_t currentCacheId = eventSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (cablingCacheId_ != currentCacheId) {
    edm::ESHandle<SiStripFedCabling> cablingHandle;
    eventSetup.get<SiStripFedCablingRcd>().get(cablingHandle);
    cabling_ = cablingHandle.product();
    cablingCacheId_ = currentCacheId;
  }
}

void SiStripFEDCheckPlugin::fillPresent(unsigned int fedId, bool present)
{
  if (present) {
    if (updateFrequency_) fedsPresentBinContents_[fedId]++;
    else fedsPresent_->Fill(fedId);
  }
}

void SiStripFEDCheckPlugin::fillFatalError(unsigned int fedId, bool fatalError)
{
  if (updateFrequency_) {
    if (fatalError) fedFatalErrorBinContents_[fedId]++;
  } else {
    //fedFatalErrors_->Fill( fatalError ? 1 : 0 );
    if (fatalError) fedFatalErrors_->Fill(fedId);
  }
}

void SiStripFEDCheckPlugin::fillNonFatalError(unsigned int fedId, float nonFatalError)
{
  if (updateFrequency_) {
    if (nonFatalError>0) fedNonFatalErrorBinContents_[fedId]++;//nonFatalError;
  } else {
    if (nonFatalError>0) fedNonFatalErrors_->Fill(fedId);
  }
}

void SiStripFEDCheckPlugin::doUpdateIfNeeded()
{
  eventCount_++;
  if (updateFrequency_ && (eventCount_%updateFrequency_ == 0)) {
    updateHistograms();
  }
}

void SiStripFEDCheckPlugin::updateHistograms()
{
  //if the cache is not being used then do nothing
  if (!updateFrequency_) return;
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  unsigned int entriesFedsPresent = 0;
  unsigned int entriesFatalErrors = 0;
  unsigned int entriesNonFatalErrors = 0;
  for (unsigned int fedId = siStripFedIdMin, bin = 1; fedId < siStripFedIdMax+1; fedId++, bin++) {
    unsigned int fedsPresentBin = fedsPresentBinContents_[fedId];
    fedsPresent_->getTH1()->SetBinContent(bin,fedsPresentBin);
    entriesFedsPresent += fedsPresentBin;
    unsigned int fedFatalErrorsBin = fedFatalErrorBinContents_[fedId];
    fedFatalErrors_->getTH1()->SetBinContent(bin,fedFatalErrorsBin);
    entriesFatalErrors += fedFatalErrorsBin;
    unsigned int fedNonFatalErrorsBin = fedNonFatalErrorBinContents_[fedId];
    fedNonFatalErrors_->getTH1()->SetBinContent(bin,fedNonFatalErrorsBin);
    entriesNonFatalErrors += fedNonFatalErrorsBin;
  }
  fedsPresent_->getTH1()->SetEntries(entriesFedsPresent);
  fedFatalErrors_->getTH1()->SetEntries(entriesFatalErrors);
  fedNonFatalErrors_->getTH1()->SetEntries(entriesNonFatalErrors);
}

//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDCheckPlugin);
