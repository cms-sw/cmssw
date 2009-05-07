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
// $Id: SiStripFEDDataCheck.cc,v 1.5 2008/11/04 09:55:11 nc302 Exp $
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <memory>

//
// Class decleration
//

class SiStripFEDCheckPlugin : public edm::EDAnalyzer
{
 public:
  explicit SiStripFEDCheckPlugin(const edm::ParameterSet&);
  ~SiStripFEDCheckPlugin();
 private:
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  
  bool hasFatalError(const FEDRawData& fedData, unsigned int fedId) const;
  bool hasNonFatalError(const FEDRawData& fedData, unsigned int fedId) const;
  void updateCabling(const edm::EventSetup& eventSetup);
  
  inline void fillPresent(unsigned int fedId, bool present);
  inline void fillFatalError(unsigned int fedId, bool fatalError);
  inline void fillNonFatalError(unsigned int fedId, bool nonFatalError);
  
  void doUpdateIfNeeded();
  void updateHistograms();
  
  
  edm::InputTag rawDataTag_;
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
  : rawDataTag_(iConfig.getUntrackedParameter<edm::InputTag>("RawDataTag",edm::InputTag("source",""))),
    dirName_(iConfig.getUntrackedParameter<std::string>("DirName","SiStrip/FEDIntegrity/")),
    printDebug_(iConfig.getUntrackedParameter<bool>("PrintDebugMessages",false)),
    writeDQMStore_(iConfig.getUntrackedParameter<bool>("WriteDQMStore",false)),
    updateFrequency_(iConfig.getUntrackedParameter<unsigned int>("HistogramUpdateFrequency",0)),
    fedsPresentBinContents_(FEDNumbering::getSiStripFEDIds().second+1,0),
    fedFatalErrorBinContents_(FEDNumbering::getSiStripFEDIds().second+1,0),
    fedNonFatalErrorBinContents_(FEDNumbering::getSiStripFEDIds().second+1,0),
    eventCount_(0),
    doPayloadChecks_(iConfig.getUntrackedParameter<bool>("DoPayloadChecks",true)),
    checkChannelLengths_(iConfig.getUntrackedParameter<bool>("CheckChannelLengths",true)),
    checkPacketCodes_(iConfig.getUntrackedParameter<bool>("CheckChannelPacketCodes",true)),
    checkFELengths_(iConfig.getUntrackedParameter<bool>("CheckFELengths",true)),
    checkChannelStatusBits_(iConfig.getUntrackedParameter<bool>("CheckChannelStatus",true)),
    cablingCacheId_(0)
{
  if (!doPayloadChecks_ && (checkChannelLengths_ || checkPacketCodes_ || checkFELengths_ || checkChannelStatusBits_) ) {
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
  //update cabling
  updateCabling(iSetup);
  
  //get raw data
  edm::Handle<FEDRawDataCollection> rawDataCollectionHandle;
  iEvent.getByLabel(rawDataTag_,rawDataCollectionHandle);
  const FEDRawDataCollection& rawDataCollection = *rawDataCollectionHandle;
  
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::getSiStripFEDIds().first;
  const unsigned int siStripFedIdMax = FEDNumbering::getSiStripFEDIds().second;
  
  //loop over siStrip FED IDs
  for (unsigned int fedId = siStripFedIdMin; fedId <= siStripFedIdMax; fedId++) {
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);
    //check data exists
    if (!fedData.size() || !fedData.data()) {
      fillPresent(fedId,0);
      continue;
    }
    //fill buffer present histogram
    fillPresent(fedId,1);
    //check for fatal errors
    if (hasFatalError(fedData,fedId)) {
      fillFatalError(fedId,1);
    } else {
      fillFatalError(fedId,0);
      //fill non-fatal errors histogram if there were no fatal errors
      fillNonFatalError(fedId,hasNonFatalError(fedData,fedId) ? 1 : 0);
    }
  }//loop over FED IDs
  
  //update histograms if needed
  doUpdateIfNeeded();
}

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripFEDCheckPlugin::beginJob(const edm::EventSetup&)
{
  //get FED IDs
  const FEDNumbering numbering;
  const unsigned int siStripFedIdMin = numbering.getSiStripFEDIds().first;
  const unsigned int siStripFedIdMax = numbering.getSiStripFEDIds().second;
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

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripFEDCheckPlugin::endJob()
{
  updateHistograms();
  if (writeDQMStore_) dqm_->save("DQMStore.root");
}


bool SiStripFEDCheckPlugin::hasFatalError(const FEDRawData& fedData, unsigned int fedId) const
{
  bool fatalError = false;
  //first build a base buffer object to do basic checks (without checking channel data)
  const sistrip::FEDBufferBase buffer(fedData.data(),fedData.size(),true);
  //check for errors signaled in DAQ header and trailer and that length is consistent with buffer length
  if (!buffer.doDAQHeaderAndTrailerChecks()) fatalError = true;
  //check that buffer format byte is valid
  if (!buffer.checkBufferFormat()) fatalError = true;
  //check CRC
  if (!buffer.checkCRC()) fatalError = true;
  //if there was an error then provide info
  if (fatalError) {
    if (printDebug_) {
      edm::LogInfo("SiStripFEDCheck") << "Fatal error with FED ID " << fedId << ". Check summary: " 
                                      << std::endl << buffer.checkSummary() << std::endl;
      std::stringstream ss;
      buffer.dump(ss);
      edm::LogInfo("SiStripFEDCheck") << ss.str();
    }
    return true;
  } else {
    return false;
  }
}

bool SiStripFEDCheckPlugin::hasNonFatalError(const FEDRawData& fedData, unsigned int fedId) const
{
  const sistrip::FEDBufferBase baseBuffer(fedData.data(),fedData.size(),true);
  if (!baseBuffer.doTrackerSpecialHeaderChecks()) {
    if (printDebug_) {
      edm::LogInfo("SiStripFEDCheck") << "Error with header for FED ID " << fedId << ". Check summary: "
                                      << std::endl << baseBuffer.checkSummary() << std::endl;
      std::stringstream ss;
      baseBuffer.dump(ss);
      edm::LogInfo("SiStripFEDCheck") << ss.str();
    }
    return true;
  }
  if (doPayloadChecks_) {
    //check that channels can all be found in buffer
    std::auto_ptr<const sistrip::FEDBuffer> pBuffer;
    try {
      pBuffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size()));
    } catch (const cms::Exception& e) {
      pBuffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size(),true));
      if (printDebug_) {
        edm::LogInfo("SiStripFEDCheck") << "Error constructing buffer object for FED ID " << fedId
                                        << std::endl << e.what() << std::endl << "Check summary: "
                                        << std::endl << pBuffer->checkSummary() << std::endl;
        std::stringstream ss;
        pBuffer->dump(ss);
        edm::LogInfo("SiStripFEDCheck") << ss.str();
      }
      return true;
    }
    //check that all fields in buffer are valid and that there are no problems with data
    bool channelLengthsOK = checkChannelLengths_ ? pBuffer->checkChannelLengthsMatchBufferLength() : true;
    bool channelPacketCodesOK = checkPacketCodes_ ? pBuffer->checkChannelPacketCodes() : true;
    bool feLengthsOK = checkFELengths_ ? pBuffer->checkFEUnitLengths() : true;
    if (!pBuffer->doChecks() ||
        !channelLengthsOK ||
        !channelPacketCodesOK ||
        !feLengthsOK ) {
      if (printDebug_) {
        edm::LogInfo("SiStripFEDCheck") << "Error with FED ID " << fedId << ". Check summary: "
                                        << std::endl << pBuffer->checkSummary() << std::endl;
        std::stringstream ss;
        pBuffer->dump(ss);
        edm::LogInfo("SiStripFEDCheck") << ss.str();
      }
      return true;
    }
    //check that channels in cabling have no bad status bits and are enabled
    if (checkChannelStatusBits_) {
      for (unsigned int c = 0; c < sistrip::FEDCH_PER_FED; c++) {
        if (!cabling_->connection(fedId,c).isConnected()) continue;
        else if (!pBuffer->channelGood(c)) {
          if (printDebug_) {
            edm::LogInfo("SiStripFEDCheck") << "Error with FED ID " << fedId << " channel " << c << ". Check summary: "
                                            << std::endl << pBuffer->checkSummary() << std::endl;
            std::stringstream ss;
            pBuffer->dump(ss);
            edm::LogInfo("SiStripFEDCheck") << ss.str();
          }
          return true;
        }
      }
    }
  }
  //if the checks above all passed then there are no errors
  return false;
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
    fedFatalErrors_->Fill( fatalError ? 1 : 0 );
  }
}

void SiStripFEDCheckPlugin::fillNonFatalError(unsigned int fedId, bool nonFatalError)
{
  if (updateFrequency_) {
    if (nonFatalError) fedNonFatalErrorBinContents_[fedId]++;
  } else {
    fedNonFatalErrors_->Fill( nonFatalError ? 1 : 0 );
  }
}

void SiStripFEDCheckPlugin::doUpdateIfNeeded()
{
  eventCount_++;
  if (updateFrequency_ && !(eventCount_%updateFrequency_)) {
    updateHistograms();
  }
}

void SiStripFEDCheckPlugin::updateHistograms()
{
  //if the cache is not being used then do nothing
  if (!updateFrequency_) return;
  const unsigned int siStripFedIdMin = FEDNumbering::getSiStripFEDIds().first;
  const unsigned int siStripFedIdMax = FEDNumbering::getSiStripFEDIds().second;
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
    fedNonFatalErrors_->getTH1()->SetBinContent(bin,fedFatalErrorsBin);
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
