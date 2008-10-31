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
// $Id: SiStripFEDDataCheck.cc,v 1.2 2008/10/20 13:13:45 nc302 Exp $
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
  
  edm::InputTag rawDataTag_;
  std::string folderName_;
  bool printDebug_;
  bool writeDQMStore_;
  DQMStore* dqm_;
  MonitorElement* fedsPresent_;
  MonitorElement* fedFatalErrors_;
  MonitorElement* fedNonFatalErrors_;
  uint32_t cablingCacheId_;
  const SiStripFedCabling* cabling_;
};


//
// Constructors and destructor
//

SiStripFEDCheckPlugin::SiStripFEDCheckPlugin(const edm::ParameterSet& iConfig)
  : rawDataTag_(iConfig.getUntrackedParameter<edm::InputTag>("RawDataTag")),
    folderName_(iConfig.getUntrackedParameter<std::string>("FolderName")),
    printDebug_(iConfig.getUntrackedParameter<bool>("PrintDebugMessages")),
    writeDQMStore_(iConfig.getUntrackedParameter<bool>("WriteDQMStore")),
    cablingCacheId_(0)
{
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
  const FEDNumbering numbering;
  const unsigned int siStripFedIdMin = numbering.getSiStripFEDIds().first;
  const unsigned int siStripFedIdMax = numbering.getSiStripFEDIds().second;
  
  //loop over siStrip FED IDs
  for (unsigned int fedId = siStripFedIdMin; fedId <= siStripFedIdMax; fedId++) {
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);
    //check data exists
    if (!fedData.size() || !fedData.data()) continue;
    //fill buffer present histogram
    fedsPresent_->Fill(fedId);
    //check for fatal errors
    if (hasFatalError(fedData,fedId)) {
      fedFatalErrors_->Fill(fedId,1);
    } else {
      fedFatalErrors_->Fill(fedId,0);
      //fill non-fatal errors histogram if there were no fatal errors
      fedNonFatalErrors_->Fill(fedId,hasNonFatalError(fedData,fedId) ? 1 : 0);    
    }
  }//loop over FED IDs
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
  dqm_->setCurrentFolder(folderName_);
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
  if (writeDQMStore_) dqm_->save("DQMStore.root");
}


bool SiStripFEDCheckPlugin::hasFatalError(const FEDRawData& fedData, unsigned int fedId) const
{
  bool fatalError = false;
  //first build an event object to do basic checks (without checking channel data)
  const sistrip::FEDBuffer buffer(fedData.data(),fedData.size(),true);
  //check for errors signaled in DAQ header and trailer and that length is consistent with buffer length
  if (!buffer.doDAQHeaderAndTrailerChecks()) fatalError = true;
  //check that event format byte is valid
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
  //check that channels can all be found in buffer
  std::auto_ptr<const sistrip::FEDBuffer> pBuffer;
  try {
    pBuffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size()));
  } catch (const cms::Exception& e) {
    pBuffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size(),true));
    if (printDebug_) {
      edm::LogInfo("SiStripFEDCheck") << "Error constructing event buffer object for FED ID " << fedId
                                      << std::endl << e.what() << std::endl << "Check summary: "
                                      << std::endl << pBuffer->checkSummary() << std::endl;
      std::stringstream ss;
      pBuffer->dump(ss);
      edm::LogInfo("SiStripFEDCheck") << ss.str();
    }
    return true;
  }
  //check that all fields in buffer are valid and that there are no problems with data
  if (!pBuffer->doChecks()) {
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

//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDCheckPlugin);
