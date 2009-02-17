// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      SiStripFEDMonitorPlugin
// 
/**\class SiStripFEDMonitorPlugin SiStripFEDMonitor.cc DQM/SiStripMonitorHardware/plugins/SiStripFEDMonitor.cc

 Description: DQM source application to produce data integrety histograms for SiStrip data for use in HLT and Prompt reco
*/
//
// Original Author:  Nicholas Cripps
//         Created:  2008/09/16
// $Id: SiStripFEDMonitor.cc,v 1.4 2008/11/06 19:40:01 nc302 Exp $
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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include <sstream>
#include <memory>
#include <list>
#include <algorithm>

//
// Class decleration
//

class SiStripFEDMonitorPlugin : public edm::EDAnalyzer
{
 public:
  explicit SiStripFEDMonitorPlugin(const edm::ParameterSet&);
  ~SiStripFEDMonitorPlugin();
 private:
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  
  //update the cabling if necessary
  void updateCabling(const edm::EventSetup& eventSetup);
  //fill a histogram if the pointer is not NULL (ie if it has been booked)
  void fillHistogram(MonitorElement* histogram, double value);
  //book the top level histograms
  void bookTopLevelHistograms();
  //book individual FED histograms or book all FED level histograms at once
  void bookFEDHistograms(unsigned int fedId, bool fullDebugMode = false);
  void bookAllFEDHistograms();
  //return true if there were no errors at the level they are analysing
  bool analyzeFED(const FEDRawData& rawData, unsigned int fedId);
  bool analyzeFEUnits(const sistrip::FEDBufferBase* buffer, unsigned int fedId);
  bool analyzeChannels(const sistrip::FEDBuffer* buffer, unsigned int fedId,
                       std::list<unsigned int>* badChannelList,
                       std::list<unsigned int>* activeBadChannelList);
  
  edm::InputTag rawDataTag_;
  std::string folderName_;
  bool printDebug_;
  bool writeDQMStore_;
  bool fillAllHistograms_;
  bool disableGlobalExpertHistograms_;
  bool disableFEDHistograms_;
  DQMStore* dqm_;
  uint32_t cablingCacheId_;
  const SiStripFedCabling* cabling_;
  //top level histograms
  MonitorElement *anyErrors_, *anyDaqProblems_, *corruptBuffers_, *invalidBuffers_, *badIDs_, *badChannelStatusBits_, *badActiveChannelStatusBits_,
                 *badDAQCRCs_, *badFEDCRCs_, *daqProblems_, *feOverflows_, *badMajorityAddresses_;
  //FED level histograms
  std::map<unsigned int,MonitorElement*> feOverflowDetailed_, badMajorityAddressDetailed_;
  std::map<unsigned int,MonitorElement*> badStatusBitsDetailed_, apvErrorDetailed_, apvAddressErrorDetailed_, unlockedDetailed_, outOfSyncDetailed_;
  //has individual FED histogram been booked? (index is FedId)
  std::vector<bool> histosBooked_, debugHistosBooked_;
  std::vector< std::vector<bool> > activeChannels_;
};


//
// Constructors and destructor
//

SiStripFEDMonitorPlugin::SiStripFEDMonitorPlugin(const edm::ParameterSet& iConfig)
  : rawDataTag_(iConfig.getUntrackedParameter<edm::InputTag>("RawDataTag",edm::InputTag("source",""))),
    folderName_(iConfig.getUntrackedParameter<std::string>("FolderName","SiStrip/ReadoutView/FedMonitoringSummary")),
    printDebug_(iConfig.getUntrackedParameter<bool>("PrintDebugMessages",false)),
    writeDQMStore_(iConfig.getUntrackedParameter<bool>("WriteDQMStore",false)),
    fillAllHistograms_(iConfig.getUntrackedParameter<bool>("FillAllHistograms",true)),
    disableGlobalExpertHistograms_(iConfig.getUntrackedParameter<bool>("DisableGlobalExpertHistograms",true)),
    disableFEDHistograms_(iConfig.getUntrackedParameter<bool>("DisableFEDHistograms",true)),
    cablingCacheId_(0)
{
}

SiStripFEDMonitorPlugin::~SiStripFEDMonitorPlugin()
{
}


//
// Member functions
//

// ------------ method called to for each event  ------------
void
SiStripFEDMonitorPlugin::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //update cabling
  updateCabling(iSetup);
  
  //get raw data
  edm::Handle<FEDRawDataCollection> rawDataCollectionHandle;
  iEvent.getByLabel(rawDataTag_,rawDataCollectionHandle);
  const FEDRawDataCollection& rawDataCollection = *rawDataCollectionHandle;
  
  //loop over siStrip FED IDs
  for (unsigned int fedId = FEDNumbering::MINSiStripFEDID; fedId <= FEDNumbering::MAXSiStripFEDID; fedId++) {
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);
    //check data exists
    if (!fedData.size() || !fedData.data()) continue;
    bool anyErrors = !analyzeFED(fedData,fedId);
    if (anyErrors) fillHistogram(anyErrors_,fedId);
  }//loop over FED IDs
}

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripFEDMonitorPlugin::beginJob(const edm::EventSetup&)
{
  //get DQM store
  dqm_ = &(*edm::Service<DQMStore>());
  dqm_->setCurrentFolder(folderName_);
  
  bookTopLevelHistograms();
  
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  //book FED level histograms
  histosBooked_.resize(siStripFedIdMax+1,false);
  debugHistosBooked_.resize(siStripFedIdMax+1,false);
  if (fillAllHistograms_) bookAllFEDHistograms();
  //mark all channels as inactive until they have been 'locked' at least once
  activeChannels_.resize(siStripFedIdMax+1);
  for (unsigned int fedId = siStripFedIdMin; fedId <= siStripFedIdMax; fedId++) {
    activeChannels_[fedId].resize(sistrip::FEDCH_PER_FED,false);
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripFEDMonitorPlugin::endJob()
{
  if (writeDQMStore_) dqm_->save("DQMStore.root");
}

void SiStripFEDMonitorPlugin::updateCabling(const edm::EventSetup& eventSetup)
{
  uint32_t currentCacheId = eventSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (cablingCacheId_ != currentCacheId) {
    edm::ESHandle<SiStripFedCabling> cablingHandle;
    eventSetup.get<SiStripFedCablingRcd>().get(cablingHandle);
    cabling_ = cablingHandle.product();
    cablingCacheId_ = currentCacheId;
  }
}

inline void SiStripFEDMonitorPlugin::fillHistogram(MonitorElement* histogram, double value)
{
  if (histogram) histogram->Fill(value);
}

bool SiStripFEDMonitorPlugin::analyzeFED(const FEDRawData& rawData, unsigned int fedId)
{
  //try to construct the basic buffer object (do not check payload)
  //if this fails then count it as an invalid buffer and stop checks since we can't understand things like buffer ordering
  std::auto_ptr<const sistrip::FEDBufferBase> bufferBase;
  try {
    bufferBase.reset(new sistrip::FEDBufferBase(rawData.data(),rawData.size()));
  } catch (const cms::Exception& e) {
    fillHistogram(invalidBuffers_,fedId);
    fillHistogram(anyDaqProblems_,fedId);
    //don't check anything else if the buffer is invalid
    return false;
  }
  //CRC checks
  //if CRC fails then don't continue as if the buffer has been corrupted in DAQ then anything else could be invalid
  if (!bufferBase->checkNoSlinkCRCError()) {
    fillHistogram(badFEDCRCs_,fedId);
    fillHistogram(anyDaqProblems_,fedId);
    return false;
  } else if (!bufferBase->checkCRC()) {
    fillHistogram(badDAQCRCs_,fedId);
    fillHistogram(anyDaqProblems_,fedId);
    return false;
  }
  //next check that it is a SiStrip buffer
  //if not then stop checks
  if (!bufferBase->checkSourceIDs() || !bufferBase->checkNoUnexpectedSourceID()) {
    fillHistogram(badIDs_,fedId);
    fillHistogram(anyDaqProblems_,fedId);
    return false;
  } 
  //if so then do DAQ header/trailer checks
  //if these fail then buffer may be incomplete and checking contents doesn't make sense
  else if (!bufferBase->doDAQHeaderAndTrailerChecks()) {
    fillHistogram(daqProblems_,fedId);
    fillHistogram(anyDaqProblems_,fedId);
    return false;
  }
  
  bool foundError = false;
  //now do checks on header
  //check that tracker special header is consistent
  if (!bufferBase->doTrackerSpecialHeaderChecks() && bufferBase->checkNoFEOverflows()) {
    fillHistogram(invalidBuffers_,fedId);
    fillHistogram(anyDaqProblems_,fedId);
    foundError = true;
  }
  //FE unit overflows
  if (!bufferBase->checkNoFEOverflows()) { 
    foundError = true;
  }
  //if FEs overflowed or tracker special header is invalid then don't bother to check payload
  bool checkPayload = !foundError;
  bool feUnitsGood = analyzeFEUnits(bufferBase.get(),fedId);
  if (!feUnitsGood) foundError = true;
  
  //payload checks
  std::auto_ptr<const sistrip::FEDBuffer> buffer;
  std::list<unsigned int> badChannels;
  std::list<unsigned int> activeBadChannels;
  //need to construct full object to go any further
  if (checkPayload) buffer.reset(new sistrip::FEDBuffer(rawData.data(),rawData.size(),true));
  if (checkPayload) {
    //corrupt buffer checks
    if (!buffer->doCorruptBufferChecks()) {
      fillHistogram(corruptBuffers_,fedId);
      foundError = true;
    }
    //channel checks
    analyzeChannels(buffer.get(),fedId,&badChannels,&activeBadChannels);
    if (badChannels.size()) {
      fillHistogram(badChannelStatusBits_,fedId);
    }
    if (activeBadChannels.size()) {
      foundError = true;
    }
  }
  
  if (foundError && printDebug_) {
    const sistrip::FEDBufferBase* debugBuffer = NULL;
    if (buffer.get()) debugBuffer = buffer.get();
    else if (bufferBase.get()) debugBuffer = bufferBase.get();
    if (debugBuffer) {
      std::ostringstream debugStream;
      if (badChannels.size()) {
        badChannels.sort();
        debugStream << "Cabled channels which had errors: ";
        for (std::list<unsigned int>::const_iterator iBadCh = badChannels.begin(); iBadCh != badChannels.end(); iBadCh++) {
          debugStream << *iBadCh << " ";
        }
        debugStream << std::endl;
      }
      if (activeBadChannels.size()) {
        activeBadChannels.sort();
        debugStream << "Active (have been unlocked in at least one event) cabled channels which had errors: ";
        for (std::list<unsigned int>::const_iterator iBadCh = activeBadChannels.begin(); iBadCh != activeBadChannels.end(); iBadCh++) {
          debugStream << *iBadCh << " ";
        }
        debugStream << std::endl;
      }
      debugStream << (*debugBuffer) << std::endl;
      debugBuffer->dump(debugStream);
      debugStream << std::endl;
      edm::LogInfo("SiStripMonitorHardware") << "Errors found in FED " << fedId;
      edm::LogVerbatim("SiStripMonitorHardware") << debugStream.str();
    }
  }
  
  return !foundError;
}

bool SiStripFEDMonitorPlugin::analyzeFEUnits(const sistrip::FEDBufferBase* buffer, unsigned int fedId)
{
  bool foundOverflow = false;
  bool foundBadMajority = false;
  for (unsigned int iFE = 0; iFE < sistrip::FEUNITS_PER_FED; iFE++) {
    if (buffer->feOverflow(iFE)) {
      bookFEDHistograms(fedId);
      fillHistogram(feOverflowDetailed_[fedId],iFE);
      foundOverflow = true;
      //if FE overflowed then address isn't valid
      continue;
    }
    if (!buffer->feEnabled(iFE)) continue;
    if (buffer->majorityAddressErrorForFEUnit(iFE)) {
      bookFEDHistograms(fedId);
      fillHistogram(badMajorityAddressDetailed_[fedId],iFE);
      foundBadMajority = true;
    }
  }
  if (foundOverflow) {
    fillHistogram(feOverflows_,fedId);
  }
  if (foundBadMajority) {
    fillHistogram(badMajorityAddresses_,fedId);
  }
  bool foundError = foundOverflow || foundBadMajority;
  return !foundError;
}

bool SiStripFEDMonitorPlugin::analyzeChannels(const sistrip::FEDBuffer* buffer, unsigned int fedId,
                                              std::list<unsigned int>* badChannelList,
                                              std::list<unsigned int>* activeBadChannelList)
{
  bool foundError = false;
  bool filledBadChannel = false;
  const sistrip::FEDFEHeader* header = buffer->feHeader();
  const sistrip::FEDFullDebugHeader* debugHeader = dynamic_cast<const sistrip::FEDFullDebugHeader*>(header);
  for (unsigned int iCh = 0; iCh < sistrip::FEDCH_PER_FED; iCh++) {
    if (!cabling_->connection(fedId,iCh).isConnected()) continue;
    if (debugHeader) {
      if (!debugHeader->unlocked(iCh)) activeChannels_[fedId][iCh] = true;
    }
    bool channelWasBad = false;
    for (unsigned int iAPV = 0; iAPV < 2; iAPV++) {
      if (!header->checkStatusBits(iCh,iAPV)) {
        bookFEDHistograms(fedId,debugHeader);
        fillHistogram(badStatusBitsDetailed_[fedId],iCh*2+iAPV);
        foundError = true;
        channelWasBad = true;
      }
    }
    //add channel to bad channel list
    if (channelWasBad && badChannelList) badChannelList->push_back(iCh);
    if (channelWasBad && activeChannels_[fedId][iCh] && activeBadChannelList) activeBadChannelList->push_back(iCh);
    //fill histogram for active channels
    if (channelWasBad && activeChannels_[fedId][iCh] && !filledBadChannel) {
      fillHistogram(badActiveChannelStatusBits_,fedId);
      filledBadChannel = true;
    }
  }
  if (debugHeader) {
    for (unsigned int iCh = 0; iCh < sistrip::FEDCH_PER_FED; iCh++) {
      for (unsigned int iAPV = 0; iAPV < 2; iAPV++) {
        if (debugHeader->apvError(iCh,iAPV)) {
          bookFEDHistograms(fedId,debugHeader);
          fillHistogram(apvErrorDetailed_[fedId],iCh*2+iAPV);
        }
        if (debugHeader->apvAddressError(iCh,iAPV)) {
          bookFEDHistograms(fedId,debugHeader);
          fillHistogram(apvAddressErrorDetailed_[fedId],iCh*2+iAPV);
        }
      }
      if (debugHeader->unlocked(iCh)) {
        bookFEDHistograms(fedId,debugHeader);
        fillHistogram(unlockedDetailed_[fedId],iCh);
      }
      if (debugHeader->outOfSync(iCh)) {
        bookFEDHistograms(fedId,debugHeader);
        fillHistogram(outOfSyncDetailed_[fedId],iCh);
      }
    }
  }
  return !foundError;
}

void SiStripFEDMonitorPlugin::bookFEDHistograms(unsigned int fedId, bool fullDebugMode)
{
  if (disableFEDHistograms_ && !fillAllHistograms_) return;
  if (!histosBooked_[fedId]) {
    SiStripFedKey fedKey(fedId,0,0,0);
    dqm_->setCurrentFolder(fedKey.path());
    std::stringstream fedIdStream;
    fedIdStream << fedId;
    feOverflowDetailed_[fedId] = dqm_->book1D("FEOverflowsForFED"+fedIdStream.str(),
                                              "FE overflows per FE unit for FED ID "+fedIdStream.str(),
                                              sistrip::FEUNITS_PER_FED,0,sistrip::FEUNITS_PER_FED);
    feOverflowDetailed_[fedId]->setAxisTitle("FE-Index",1);
    badMajorityAddressDetailed_[fedId] = dqm_->book1D("BadMajorityAddressesForFED"+fedIdStream.str(),
                                                      "Bad majority APV addresses per FE unit for FED ID "+fedIdStream.str(),
                                                      sistrip::FEUNITS_PER_FED,0,sistrip::FEUNITS_PER_FED);
    badMajorityAddressDetailed_[fedId]->setAxisTitle("FE-Index",1);
    badStatusBitsDetailed_[fedId] = dqm_->book1D("BadAPVStatusBitsForFED"+fedIdStream.str(),
                                                 "Bad apv status bits for FED ID "+fedIdStream.str(),
                                                 sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED);
    badStatusBitsDetailed_[fedId]->setAxisTitle("APV-Index",1);
    histosBooked_[fedId] = true;
  }
  if (fullDebugMode && !debugHistosBooked_[fedId]) {
    SiStripFedKey fedKey(fedId,0,0,0);
    dqm_->setCurrentFolder(fedKey.path());
    std::stringstream fedIdStream;
    fedIdStream << fedId;
    apvErrorDetailed_[fedId] = dqm_->book1D("APVErrorBitsForFED"+fedIdStream.str(),
                                            "APV errors for FED ID "+fedIdStream.str(),
                                            sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED);
    apvErrorDetailed_[fedId]->setAxisTitle("APV-Index",1);
    apvAddressErrorDetailed_[fedId] = dqm_->book1D("APVAddressErrorBitsForFED"+fedIdStream.str(),
                                                   "Wrong APV address errors for FED ID "+fedIdStream.str(),
                                                   sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED);
    apvAddressErrorDetailed_[fedId]->setAxisTitle("APV-Index",1);
    unlockedDetailed_[fedId] = dqm_->book1D("UnlockedBitsForFED"+fedIdStream.str(),
                                            "Unlocked channels for FED ID "+fedIdStream.str(),
                                            sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED);
    unlockedDetailed_[fedId]->setAxisTitle("Channel-Index",1);
    outOfSyncDetailed_[fedId] = dqm_->book1D("OOSBitsForFED"+fedIdStream.str(),
                                             "Out of sync channels for FED ID "+fedIdStream.str(),
                                             sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED);
    outOfSyncDetailed_[fedId]->setAxisTitle("Channel-Index",1);
    debugHistosBooked_[fedId] = true;
  }
}

void SiStripFEDMonitorPlugin::bookAllFEDHistograms()
{
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  //book them
  for (unsigned int iFed = siStripFedIdMin; iFed <= siStripFedIdMax; iFed++) {
    bookFEDHistograms(iFed,true);
  }
}

void SiStripFEDMonitorPlugin::bookTopLevelHistograms()
{
  //get FED IDs
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  //book histos
  anyErrors_ = dqm_->book1D("AnyErrors",
                            "Number of buffers with any error per FED",
                            siStripFedIdMax-siStripFedIdMin+1,
                            siStripFedIdMin-0.5,siStripFedIdMax+0.5);
  anyErrors_->setAxisTitle("FED-ID",1);
  corruptBuffers_ = dqm_->book1D("CorruptBuffers",
                                 "Number of corrupt FED buffers per FED",
                                 siStripFedIdMax-siStripFedIdMin+1,
                                 siStripFedIdMin-0.5,siStripFedIdMax+0.5);
  corruptBuffers_->setAxisTitle("FED-ID",1);
  invalidBuffers_ = dqm_->book1D("InvalidBuffers",
                                 "Number of invalid FED buffers per FED",
                                 siStripFedIdMax-siStripFedIdMin+1,
                                 siStripFedIdMin-0.5,siStripFedIdMax+0.5);
  invalidBuffers_->setAxisTitle("FED-ID",1);
  anyDaqProblems_ = dqm_->book1D("AnyDAQProblems",
                                 "Number of buffers with any problems flagged in DAQ header (including CRC)",
                                 siStripFedIdMax-siStripFedIdMin+1,
                                 siStripFedIdMin-0.5,siStripFedIdMax+0.5);
  anyDaqProblems_->setAxisTitle("FED-ID",1);
  badIDs_ = dqm_->book1D("BadIDs",
                         "Number of buffers with non-SiStrip source IDs in DAQ header",
                         siStripFedIdMax-siStripFedIdMin+1,
                         siStripFedIdMin-0.5,siStripFedIdMax+0.5);
  badIDs_->setAxisTitle("FED-ID",1);
  badChannelStatusBits_ = dqm_->book1D("BadChannelStatusBits",
                                       "Number of buffers with one or more enabled channel with bad status bits",
                                       siStripFedIdMax-siStripFedIdMin+1,
                                       siStripFedIdMin-0.5,siStripFedIdMax+0.5);
  badChannelStatusBits_->setAxisTitle("FED-ID",1);
  badActiveChannelStatusBits_ = dqm_->book1D("BadActiveChannelStatusBits",
                                             "Number of buffers with one or more active channel with bad status bits",
                                             siStripFedIdMax-siStripFedIdMin+1,
                                             siStripFedIdMin-0.5,siStripFedIdMax+0.5);
  badActiveChannelStatusBits_->setAxisTitle("FED-ID",1);
  if (!disableGlobalExpertHistograms_ || fillAllHistograms_) {
    badDAQCRCs_ = dqm_->book1D("BadDAQCRCs",
                               "Number of buffers with bad CRCs from the DAQ",
                               siStripFedIdMax-siStripFedIdMin+1,
                               siStripFedIdMin-0.5,siStripFedIdMax+0.5);
    badDAQCRCs_->setAxisTitle("FED-ID",1);
    badFEDCRCs_ = dqm_->book1D("BadFEDCRCs",
                               "Number of buffers with bad CRCs from the FED",
                               siStripFedIdMax-siStripFedIdMin+1,
                               siStripFedIdMin-0.5,siStripFedIdMax+0.5);
    badFEDCRCs_->setAxisTitle("FED-ID",1);
    daqProblems_ = dqm_->book1D("DAQProblems",
                                "Number of buffers with (non-CRC) problems flagged in DAQ header",
                                siStripFedIdMax-siStripFedIdMin+1,
                                siStripFedIdMin-0.5,siStripFedIdMax+0.5);
    daqProblems_->setAxisTitle("FED-ID",1);
    feOverflows_ = dqm_->book1D("FEOverflows",
                                "Number of buffers with one or more FE overflow",
                                siStripFedIdMax-siStripFedIdMin+1,
                                siStripFedIdMin-0.5,siStripFedIdMax+0.5);
    feOverflows_->setAxisTitle("FED-ID",1);
    badMajorityAddresses_ = dqm_->book1D("BadMajorityAddresses",
                                         "Number of buffers with one or more FE with a bad majority APV address",
                                         siStripFedIdMax-siStripFedIdMin+1,
                                         siStripFedIdMin-0.5,siStripFedIdMax+0.5);
    badMajorityAddresses_->setAxisTitle("FED-ID",1);
  }
}

//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDMonitorPlugin);
