// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      SiStripFEDMonitorPlugin
// 
/**\class SiStripFEDMonitorPlugin SiStripFEDMonitor.cc DQM/SiStripMonitorHardware/plugins/SiStripFEDMonitor.cc

 Description: DQM source application to produce data integrety histograms for SiStrip data
*/
//
// Original Author:  Nicholas Cripps
//         Created:  2008/09/16
// $Id: SiStripFEDMonitor.cc,v 1.8 2009/03/24 14:05:22 nc302 Exp $
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
  //load the config for a histogram from PSet called <configName>HistogramConfig (writes a debug message to stream if pointer is non-NULL)
  void getConfigForHistogram(const std::string& configName, const edm::ParameterSet& psetContainingConfigPSet, std::ostringstream* pDebugStream);
  //book an individual hiostogram if enabled in config
  MonitorElement* bookHistogram(const std::string& configName, const std::string& name, const std::string& title,
                                const unsigned int nBins, const double min, const double max,
                                const std::string& xAxisTitle);
  //same but using binning from config
  MonitorElement* bookHistogram(const std::string& configName, const std::string& name, const std::string& title, const std::string& xAxisTitle);
  //return true if there were no errors at the level they are analysing
  //ie analyze FED returns true if there were no FED level errors which prevent the whole FED being unpacked
  bool analyzeFED(const FEDRawData& rawData, unsigned int fedId, unsigned int* nFEDErrors,
                  unsigned int* nDAQProblems, unsigned int* nFEDsWithFEProblems, unsigned int* nCorruptBuffers, unsigned int* nBadActiveChannels,
                  unsigned int* nFEDsWithFEOverflows, unsigned int* nFEDsWithFEBadMajorityAddresses,
                  unsigned int* nFEDsWithMissingFEs);
  bool analyzeFEUnits(const sistrip::FEDBuffer* buffer, unsigned int fedId,
                      unsigned int* nFEOverflows, unsigned int* nFEBadMajorityAddresses, unsigned int* nFEMissing);
  bool analyzeChannels(const sistrip::FEDBuffer* buffer, unsigned int fedId,
                       std::list<unsigned int>* badChannelList,
                       std::list<unsigned int>* activeBadChannelList);
  
  struct HistogramConfig {
    bool enabled;
    unsigned int nBins;
    double min;
    double max;
  };
  
  //tag of FEDRawData collection
  edm::InputTag rawDataTag_;
  //folder name for histograms in DQMStore
  std::string folderName_;
  //book detailed histograms even if they will be empty (for merging)
  bool fillAllDetailedHistograms_;
  //print debug messages when problems are found
  bool printDebug_;
  //write the DQMStore to a root file at the end of the job
  bool writeDQMStore_;
  std::string dqmStoreFileName_;
  //the DQMStore
  DQMStore* dqm_;
  //FED cabling
  uint32_t cablingCacheId_;
  const SiStripFedCabling* cabling_;
  //config for histograms (enabled? bins)
  std::map<std::string,HistogramConfig> histogramConfig_;
  //counting histograms (histogram of number of problems per event)
  MonitorElement *nFEDErrors_, *nFEDDAQProblems_, *nFEDsWithFEProblems_, *nFEDCorruptBuffers_, *nBadActiveChannelStatusBits_,
                 *nFEDsWithFEOverflows_, *nFEDsWithFEBadMajorityAddresses_, *nFEDsWithMissingFEs_;
  //top level histograms
  MonitorElement *anyFEDErrors_, *anyDAQProblems_, *corruptBuffers_, *invalidBuffers_, *badIDs_, *badChannelStatusBits_, *badActiveChannelStatusBits_,
                 *badDAQCRCs_, *badFEDCRCs_, *badDAQPacket, *dataMissing_, *dataPresent_, *feOverflows_, *badMajorityAddresses_, *feMissing_, *anyFEProblems_;
  //FED level histograms
  std::map<unsigned int,MonitorElement*> feOverflowDetailed_, badMajorityAddressDetailed_, feMissingDetailed_;
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
    folderName_(iConfig.getUntrackedParameter<std::string>("HistogramFolderName","SiStrip/ReadoutView/FedMonitoringSummary")),
    fillAllDetailedHistograms_(iConfig.getUntrackedParameter<bool>("FillAllDetailedHistograms",true)),
    printDebug_(iConfig.getUntrackedParameter<bool>("PrintDebugMessages",false)),
    writeDQMStore_(iConfig.getUntrackedParameter<bool>("WriteDQMStore",false)),
    dqmStoreFileName_(iConfig.getUntrackedParameter<std::string>("DQMStoreFileName","DQMStore.root")),
    cablingCacheId_(0)
{
  //print config to debug log
  std::ostringstream debugStream;
  if (printDebug_) {
    debugStream << "Configuration for SiStripFEDMonitorPlugin: " << std::endl
                << "\tRawDataTag: " << rawDataTag_ << std::endl
                << "\tHistogramFolderName: " << folderName_ << std::endl
                << "\tFillAllDetailedHistograms? " << (fillAllDetailedHistograms_ ? "yes" : "no") << std::endl
                << "\tPrintDebugMessages? " << (printDebug_ ? "yes" : "no") << std::endl
                << "\tWriteDQMStore? " << (writeDQMStore_ ? "yes" : "no") << std::endl;
    if (writeDQMStore_) debugStream << "\tDQMStoreFileName: " << dqmStoreFileName_ << std::endl;
  }
  
  //don;t generate debug mesages if debug is disabled
  std::ostringstream* pDebugStream = (printDebug_ ? &debugStream : NULL);
  
  getConfigForHistogram("DataPresent",iConfig,pDebugStream);
  getConfigForHistogram("AnyFEDErrors",iConfig,pDebugStream);
  getConfigForHistogram("AnyDAQProblems",iConfig,pDebugStream);
  getConfigForHistogram("AnyFEProblems",iConfig,pDebugStream);
  getConfigForHistogram("CorruptBuffers",iConfig,pDebugStream);
  getConfigForHistogram("BadChannelStatusBits",iConfig,pDebugStream);
  getConfigForHistogram("BadActiveChannelStatusBits",iConfig,pDebugStream);
  
  getConfigForHistogram("FEOverflows",iConfig,pDebugStream);
  getConfigForHistogram("FEMissing",iConfig,pDebugStream);
  getConfigForHistogram("BadMajorityAddresses",iConfig,pDebugStream);
  
  getConfigForHistogram("DataMissing",iConfig,pDebugStream);
  getConfigForHistogram("BadIDs",iConfig,pDebugStream);
  getConfigForHistogram("BadDAQPacket",iConfig,pDebugStream);
  getConfigForHistogram("InvalidBuffers",iConfig,pDebugStream);
  getConfigForHistogram("BadDAQCRCs",iConfig,pDebugStream);
  getConfigForHistogram("BadFEDCRCs",iConfig,pDebugStream);
  
  getConfigForHistogram("FEOverflowsDetailed",iConfig,pDebugStream);
  getConfigForHistogram("FEMissingDetailed",iConfig,pDebugStream);
  getConfigForHistogram("BadMajorityAddressesDetailed",iConfig,pDebugStream);
  getConfigForHistogram("BadAPVStatusBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram("APVErrorBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram("APVAddressErrorBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram("UnlockedBitsDetailed",iConfig,pDebugStream);
  getConfigForHistogram("OOSBitsDetailed",iConfig,pDebugStream);
  
  getConfigForHistogram("nFEDErrors",iConfig,pDebugStream);
  getConfigForHistogram("nFEDDAQProblems",iConfig,pDebugStream);
  getConfigForHistogram("nFEDsWithFEProblems",iConfig,pDebugStream);
  getConfigForHistogram("nFEDCorruptBuffers",iConfig,pDebugStream);
  getConfigForHistogram("nBadActiveChannelStatusBits",iConfig,pDebugStream);
  getConfigForHistogram("nFEDsWithFEOverflows",iConfig,pDebugStream);
  getConfigForHistogram("nFEDsWithMissingFEs",iConfig,pDebugStream);
  getConfigForHistogram("nFEDsWithFEBadMajorityAddresses",iConfig,pDebugStream);
  
  if (printDebug_) LogTrace("SiStripMonitorHardware") << debugStream.str();
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
  
  //counters
  unsigned int nFEDErrors = 0;
  unsigned int nDAQProblems = 0;
  unsigned int nFEProblems = 0;
  unsigned int nCorruptBuffers = 0;
  unsigned int nBadActiveChannels = 0;
  unsigned int nFEDsWithFEOverflows = 0;
  unsigned int nFEDsWithFEBadMajorityAddresses = 0;
  unsigned int nFEDsWithMissingFEs = 0;
  
  //loop over siStrip FED IDs
  for (unsigned int fedId = FEDNumbering::MINSiStripFEDID; fedId <= FEDNumbering::MAXSiStripFEDID; fedId++) {
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);
    //check data exists
    if (!fedData.size() || !fedData.data()) {
      bool fedHasCabledChannels = false;
      for (unsigned int iCh = 0; iCh < sistrip::FEDCH_PER_FED; iCh++) {
        if (cabling_->connection(fedId,iCh).isConnected()) {
          fedHasCabledChannels = true;
          break;
        }
      }
      if (fedHasCabledChannels) {
        fillHistogram(dataMissing_,fedId);
        fillHistogram(anyDAQProblems_,fedId);
      }
      continue;
    } else {
      fillHistogram(dataPresent_,fedId);
    }
    //check for problems and fill detailed histograms
    const bool anyFEDErrors = !analyzeFED(fedData,fedId,&nFEDErrors,&nDAQProblems,&nFEProblems,&nCorruptBuffers,&nBadActiveChannels,
                                          &nFEDsWithFEOverflows,&nFEDsWithFEBadMajorityAddresses,&nFEDsWithMissingFEs);
    if (anyFEDErrors) fillHistogram(anyFEDErrors_,fedId);
  }//loop over FED IDs
  
  //fill count histograms
  fillHistogram(nFEDErrors_,nFEDErrors);
  fillHistogram(nFEDDAQProblems_,nDAQProblems);
  fillHistogram(nFEDsWithFEProblems_,nFEProblems);
  fillHistogram(nFEDCorruptBuffers_,nCorruptBuffers);
  fillHistogram(nFEDsWithFEOverflows_,nFEDsWithFEOverflows);
  fillHistogram(nFEDsWithFEBadMajorityAddresses_,nFEDsWithFEBadMajorityAddresses);
  fillHistogram(nFEDsWithMissingFEs_,nFEDsWithMissingFEs);
  fillHistogram(nBadActiveChannelStatusBits_,nBadActiveChannels);
  
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
  if (fillAllDetailedHistograms_) bookAllFEDHistograms();
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
  if (writeDQMStore_) dqm_->save(dqmStoreFileName_);
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

bool SiStripFEDMonitorPlugin::analyzeFED(const FEDRawData& rawData, unsigned int fedId, unsigned int* nFEDErrors,
                                         unsigned int* nDAQProblems, unsigned int* nFEDsWithFEProblems, unsigned int* nCorruptBuffers,
                                         unsigned int* nBadActiveChannels,
                                         unsigned int* nFEDsWithFEOverflows, unsigned int* nFEDsWithFEBadMajorityAddresses, unsigned int* nFEDsWithMissingFEs)
{
  //try to construct the basic buffer object (do not check payload)
  //if this fails then count it as an invalid buffer and stop checks since we can't understand things like buffer ordering
  std::auto_ptr<const sistrip::FEDBufferBase> bufferBase;
  try {
    bufferBase.reset(new sistrip::FEDBufferBase(rawData.data(),rawData.size()));
  } catch (const cms::Exception& e) {
    fillHistogram(invalidBuffers_,fedId);
    fillHistogram(anyDAQProblems_,fedId);
    (*nDAQProblems)++;
    (*nFEDErrors)++;
    //don't check anything else if the buffer is invalid
    return false;
  }
  //CRC checks
  //if CRC fails then don't continue as if the buffer has been corrupted in DAQ then anything else could be invalid
  if (!bufferBase->checkNoSlinkCRCError()) {
    fillHistogram(badFEDCRCs_,fedId);
    fillHistogram(anyDAQProblems_,fedId);
    (*nDAQProblems)++;
    (*nFEDErrors)++;
    return false;
  } else if (!bufferBase->checkCRC()) {
    fillHistogram(badDAQCRCs_,fedId);
    fillHistogram(anyDAQProblems_,fedId);
    (*nDAQProblems)++;
    (*nFEDErrors)++;
    return false;
  }
  //next check that it is a SiStrip buffer
  //if not then stop checks
  if (!bufferBase->checkSourceIDs() || !bufferBase->checkNoUnexpectedSourceID()) {
    fillHistogram(badIDs_,fedId);
    fillHistogram(anyDAQProblems_,fedId);
    (*nDAQProblems)++;
    (*nFEDErrors)++;
    return false;
  } 
  //if so then do DAQ header/trailer checks
  //if these fail then buffer may be incomplete and checking contents doesn't make sense
  else if (!bufferBase->doDAQHeaderAndTrailerChecks()) {
    fillHistogram(badDAQPacket,fedId);
    fillHistogram(anyDAQProblems_,fedId);
    (*nDAQProblems)++;
    (*nFEDErrors)++;
    return false;
  }
  
  bool foundError = false;
  //now do checks on header
  //check that tracker special header is consistent
  if ( !(bufferBase->checkBufferFormat() && bufferBase->checkHeaderType() && bufferBase->checkReadoutMode() && bufferBase->checkAPVEAddressValid()) ) {
    fillHistogram(invalidBuffers_,fedId);
    fillHistogram(anyDAQProblems_,fedId);
    (*nDAQProblems)++;
    foundError = true;
  }
  //FE unit overflows
  if (!bufferBase->checkNoFEOverflows()) { 
    foundError = true;
  }
  const bool foundFEDLevelError = foundError;
  
  //need to construct full object to go any further
  std::auto_ptr<const sistrip::FEDBuffer> buffer;
  buffer.reset(new sistrip::FEDBuffer(rawData.data(),rawData.size(),true));
  
  //if FEs overflowed or tracker special header is invalid then don't bother to check payload
  bool checkPayload = !foundError;
  unsigned int nFEOverflows = 0;
  unsigned int nFEBadMajorityAddresses = 0;
  unsigned int nFEMissing = 0;
  bool feUnitsGood = analyzeFEUnits(buffer.get(),fedId,&nFEOverflows,&nFEBadMajorityAddresses,&nFEMissing);
  if (!feUnitsGood) foundError = true;
  if (nFEOverflows) (*nFEDsWithFEOverflows)++;
  if (nFEBadMajorityAddresses) (*nFEDsWithFEBadMajorityAddresses)++;
  if (nFEMissing) (*nFEDsWithMissingFEs)++;
  if (nFEMissing || nFEBadMajorityAddresses || nFEOverflows) (*nFEDsWithFEProblems)++;
  
  //payload checks
  std::list<unsigned int> badChannels;
  std::list<unsigned int> activeBadChannels;
  if (checkPayload) {
    //corrupt buffer checks
    if (!buffer->doCorruptBufferChecks()) {
      fillHistogram(corruptBuffers_,fedId);
      (*nCorruptBuffers)++;
      foundError = true;
    }
    //if there has been a FED error then count it then check channels
    if (foundError) (*nFEDErrors)++;
    //channel checks
    analyzeChannels(buffer.get(),fedId,&badChannels,&activeBadChannels);
    if (badChannels.size()) {
      fillHistogram(badChannelStatusBits_,fedId);
    }
    if (activeBadChannels.size()) {
      foundError = true;
      (*nBadActiveChannels) += activeBadChannels.size();
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
  
  return !foundFEDLevelError;
}

bool SiStripFEDMonitorPlugin::analyzeFEUnits(const sistrip::FEDBuffer* buffer, unsigned int fedId,
                                             unsigned int* nFEOverflows, unsigned int* nFEBadMajorityAddresses,
                                             unsigned int* nFEMissing)
{
  bool foundOverflow = false;
  bool foundBadMajority = false;
  bool foundMissing = false;
  for (unsigned int iFE = 0; iFE < sistrip::FEUNITS_PER_FED; iFE++) {
    if (buffer->feOverflow(iFE)) {
      bookFEDHistograms(fedId);
      fillHistogram(feOverflowDetailed_[fedId],iFE);
      foundOverflow = true;
      (*nFEOverflows)++;
      //if FE overflowed then address isn't valid
      continue;
    }
    if (!buffer->feEnabled(iFE)) continue;
    //check for cabled channels
    bool hasCabledChannels = false;
    for (unsigned int feUnitCh = 0; feUnitCh < sistrip::FEDCH_PER_FEUNIT; feUnitCh++) {
      if (cabling_->connection(fedId,iFE*sistrip::FEDCH_PER_FEUNIT+feUnitCh).isConnected()) {
        hasCabledChannels = true;
        break;
      }
    }
    //check for missing data
    /*if (!buffer->fePresent(iFE)) {
      if (hasCabledChannels) {
        fillHistogram(feMissingDetailed_[fedId],iFE);
        foundMissing = true;
        (*nFEMissing)++;
      }
      continue;
      }*/
    if (buffer->majorityAddressErrorForFEUnit(iFE)) {
      bookFEDHistograms(fedId);
      fillHistogram(badMajorityAddressDetailed_[fedId],iFE);
      foundBadMajority = true;
      (*nFEBadMajorityAddresses)++;
    }
  }
  if (foundOverflow) {
    fillHistogram(feOverflows_,fedId);
  }
  if (foundMissing) {
    fillHistogram(feMissing_,fedId);
  }
  if (foundBadMajority) {
    fillHistogram(badMajorityAddresses_,fedId);
  }
  bool foundError = foundOverflow || foundBadMajority || foundMissing;
  if (foundError) {
    fillHistogram(anyFEProblems_,fedId);
  }
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
    if (!buffer->feGood(iCh/sistrip::FEDCH_PER_FEUNIT)) continue;
    if (debugHeader) {
      if (!debugHeader->unlocked(iCh)) activeChannels_[fedId][iCh] = true;
    } else {
      if (header->checkChannelStatusBits(iCh)) activeChannels_[fedId][iCh] = true;
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

void SiStripFEDMonitorPlugin::getConfigForHistogram(const std::string& configName, const edm::ParameterSet& psetContainingConfigPSet,
                                                    std::ostringstream* pDebugStream)
{
  HistogramConfig config;
  const std::string psetName = configName+std::string("HistogramConfig");
  if (psetContainingConfigPSet.exists(psetName)) {
    const edm::ParameterSet& pset = psetContainingConfigPSet.getUntrackedParameter<edm::ParameterSet>(psetName);
    config.enabled = (pset.exists("Enabled") ? pset.getUntrackedParameter<bool>("Enabled") : true);
    if (config.enabled) {
      config.nBins = (pset.exists("NBins") ? pset.getUntrackedParameter<unsigned int>("NBins") : 0);
      config.min = (pset.exists("Min") ? pset.getUntrackedParameter<double>("Min") : 0);
      config.max = (pset.exists("Max") ? pset.getUntrackedParameter<double>("Max") : 0);
      if (config.nBins) {
        if (pDebugStream) (*pDebugStream) << "\tHistogram: " << configName << "\tEnabled"
                                          << "\tNBins: " << config.nBins << "\tMin: " << config.min << "\tMax: " << config.max << std::endl;
      } else {
        if (pDebugStream) (*pDebugStream) << "\tHistogram: " << configName << "\tEnabled" << std::endl;
      }
    } else {
      config.enabled = false;
      config.nBins = 0;
      config.min = config.max = 0.;
      if (pDebugStream) (*pDebugStream) << "\tHistogram: " << configName << "\tDisabled" << std::endl;
    }
  } else {
    config.enabled = false;
    config.nBins = 0;
    config.min = config.max = 0.;
    if (pDebugStream) (*pDebugStream) << "\tHistogram: " << configName << "\tDisabled" << std::endl;
  }
  histogramConfig_[configName] = config;
}

MonitorElement* SiStripFEDMonitorPlugin::bookHistogram(const std::string& configName, const std::string& name, const std::string& title,
                                                       const std::string& xAxisTitle)
{
  return bookHistogram(configName,name,title,histogramConfig_[configName].nBins,histogramConfig_[configName].min,histogramConfig_[configName].max,xAxisTitle);
}

MonitorElement* SiStripFEDMonitorPlugin::bookHistogram(const std::string& configName, const std::string& name, const std::string& title,
                                                       const unsigned int nBins, const double min, const double max,
                                                       const std::string& xAxisTitle)
{
  if (histogramConfig_[configName].enabled) {
    MonitorElement* histo = dqm_->book1D(name,title,nBins,min,max);
    histo->setAxisTitle(xAxisTitle,1);
    return histo;
  } else {
    return NULL;
  }
}

void SiStripFEDMonitorPlugin::bookFEDHistograms(unsigned int fedId, bool fullDebugMode)
{
  if (!histosBooked_[fedId]) {
    SiStripFedKey fedKey(fedId,0,0,0);
    dqm_->setCurrentFolder(fedKey.path());
    std::stringstream fedIdStream;
    fedIdStream << fedId;
    feOverflowDetailed_[fedId] = bookHistogram("FEOverflowsDetailed",
                                               "FEOverflowsForFED"+fedIdStream.str(),
                                               "FE overflows per FE unit for FED ID "+fedIdStream.str(),
                                               sistrip::FEUNITS_PER_FED,0,sistrip::FEUNITS_PER_FED,
                                               "FE-Index");
    badMajorityAddressDetailed_[fedId] = bookHistogram("BadMajorityAddressesDetailed",
                                                       "BadMajorityAddressesForFED"+fedIdStream.str(),
                                                       "Bad majority APV addresses per FE unit for FED ID "+fedIdStream.str(),
                                                       sistrip::FEUNITS_PER_FED,0,sistrip::FEUNITS_PER_FED,
                                                       "FE-Index");
    feMissingDetailed_[fedId] = bookHistogram("FEMissingDetailed",
                                              "FEMissingForFED"+fedIdStream.str(),
                                              "Buffers with FE Unit payload missing per FE unit for FED ID "+fedIdStream.str(),
                                              sistrip::FEUNITS_PER_FED,0,sistrip::FEUNITS_PER_FED,
                                              "FE-Index");
    badStatusBitsDetailed_[fedId] = bookHistogram("BadAPVStatusBitsDetailed",
                                                  "BadAPVStatusBitsForFED"+fedIdStream.str(),
                                                  "Bad apv status bits for FED ID "+fedIdStream.str(),
                                                  sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED,
                                                  "APV-Index");
    histosBooked_[fedId] = true;
  }
  if (fullDebugMode && !debugHistosBooked_[fedId]) {
    SiStripFedKey fedKey(fedId,0,0,0);
    dqm_->setCurrentFolder(fedKey.path());
    std::stringstream fedIdStream;
    fedIdStream << fedId;
    apvErrorDetailed_[fedId] = bookHistogram("APVErrorBitsDetailed",
                                             "APVErrorBitsForFED"+fedIdStream.str(),
                                             "APV errors for FED ID "+fedIdStream.str(),
                                             sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED,
                                             "APV-Index");
    apvAddressErrorDetailed_[fedId] = bookHistogram("APVAddressErrorBitsDetailed",
                                                    "APVAddressErrorBitsForFED"+fedIdStream.str(),
                                                    "Wrong APV address errors for FED ID "+fedIdStream.str(),
                                                    sistrip::APVS_PER_FED,0,sistrip::APVS_PER_FED,
                                                    "APV-Index");
    unlockedDetailed_[fedId] = bookHistogram("UnlockedBitsDetailed",
                                             "UnlockedBitsForFED"+fedIdStream.str(),
                                             "Unlocked channels for FED ID "+fedIdStream.str(),
                                             sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
                                             "Channel-Index");
    outOfSyncDetailed_[fedId] = bookHistogram("OOSBitsDetailed",
                                              "OOSBitsForFED"+fedIdStream.str(),
                                              "Out of sync channels for FED ID "+fedIdStream.str(),
                                              sistrip::FEDCH_PER_FED,0,sistrip::FEDCH_PER_FED,
                                              "Channel-Index");
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
  dataPresent_ = bookHistogram("DataPresent","DataPresent",
                               "Number of events where the data from a FED is seen",
                               siStripFedIdMax-siStripFedIdMin+1,
                               siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  dataMissing_ = bookHistogram("DataMissing","DataMissing",
                               "Number of events where the data from a FED with cabled channels is missing",
                               siStripFedIdMax-siStripFedIdMin+1,
                               siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  anyFEDErrors_ = bookHistogram("AnyFEDErrors","AnyFEDErrors",
                             "Number of buffers with any FED error (excluding bad channel status bits, FE problems except overflows) per FED",
                             siStripFedIdMax-siStripFedIdMin+1,
                             siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  corruptBuffers_ = bookHistogram("CorruptBuffers","CorruptBuffers",
                                  "Number of corrupt FED buffers per FED",
                                  siStripFedIdMax-siStripFedIdMin+1,
                                  siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  invalidBuffers_ = bookHistogram("InvalidBuffers","InvalidBuffers",
                                  "Number of invalid FED buffers per FED",
                                  siStripFedIdMax-siStripFedIdMin+1,
                                  siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  anyDAQProblems_ = bookHistogram("AnyDAQProblems","AnyDAQProblems",
                                  "Number of buffers with any problems flagged in DAQ header (including CRC)",
                                  siStripFedIdMax-siStripFedIdMin+1,
                                  siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badIDs_ = bookHistogram("BadIDs","BadIDs",
                          "Number of buffers with non-SiStrip source IDs in DAQ header",
                          siStripFedIdMax-siStripFedIdMin+1,
                          siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badChannelStatusBits_ = bookHistogram("BadChannelStatusBits","BadChannelStatusBits",
                                        "Number of buffers with one or more enabled channel with bad status bits",
                                        siStripFedIdMax-siStripFedIdMin+1,
                                        siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badActiveChannelStatusBits_ = bookHistogram("BadActiveChannelStatusBits","BadActiveChannelStatusBits",
                                              "Number of buffers with one or more active channel with bad status bits",
                                              siStripFedIdMax-siStripFedIdMin+1,
                                              siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  anyFEProblems_ = bookHistogram("AnyFEProblems","AnyFEProblems",
                                  "Number of buffers with any FE unit problems",
                                  siStripFedIdMax-siStripFedIdMin+1,
                                  siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  
  badDAQCRCs_ = bookHistogram("BadDAQCRCs","BadDAQCRCs",
                              "Number of buffers with bad CRCs from the DAQ",
                              siStripFedIdMax-siStripFedIdMin+1,
                              siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badFEDCRCs_ = bookHistogram("BadFEDCRCs","BadFEDCRCs",
                              "Number of buffers with bad CRCs from the FED",
                              siStripFedIdMax-siStripFedIdMin+1,
                              siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badDAQPacket = bookHistogram("BadDAQPacket","BadDAQPacket",
                               "Number of buffers with (non-CRC) problems flagged in DAQ header/trailer",
                               siStripFedIdMax-siStripFedIdMin+1,
                               siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  feOverflows_ = bookHistogram("FEOverflows","FEOverflows",
                               "Number of buffers with one or more FE overflow",
                               siStripFedIdMax-siStripFedIdMin+1,
                               siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  badMajorityAddresses_ = bookHistogram("BadMajorityAddresses","BadMajorityAddresses",
                                        "Number of buffers with one or more FE with a bad majority APV address",
                                        siStripFedIdMax-siStripFedIdMin+1,
                                        siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  feMissing_ = bookHistogram("FEMissing","FEMissing",
                             "Number of buffers with one or more FE unit payload missing",
                             siStripFedIdMax-siStripFedIdMin+1,
                             siStripFedIdMin-0.5,siStripFedIdMax+0.5,"FED-ID");
  
  nFEDErrors_ = bookHistogram("nFEDErrors","nFEDErrors",
                              "Number of FEDs with errors (exclusing channel status bits) per event","");
  nFEDDAQProblems_ = bookHistogram("nFEDDAQProblems","nFEDDAQProblems",
                                   "Number of FEDs with DAQ problems per event","");
  nFEDsWithFEProblems_ = bookHistogram("nFEDsWithFEProblems","nFEDsWithFEProblems",
                                       "Number of FEDs with FE problems per event","");
  nFEDCorruptBuffers_ = bookHistogram("nFEDCorruptBuffers","nFEDCorruptBuffers",
                                      "Number of FEDs with corrupt buffers per event","");
  nBadActiveChannelStatusBits_ = bookHistogram("nBadActiveChannelStatusBits","nBadActiveChannelStatusBits",
                                               "Number of active channels with bad status bits per event","");
  nFEDsWithFEOverflows_ = bookHistogram("nFEDsWithFEOverflows","nFEDsWithFEOverflows",
                                        "Number FEDs with FE units which overflowed per event","");
  nFEDsWithFEBadMajorityAddresses_ = bookHistogram("nFEDsWithFEBadMajorityAddresses","nFEDsWithFEBadMajorityAddresses",
                                                   "Number of FEDs with FE units with a bad majority address per event","");
  nFEDsWithMissingFEs_ = bookHistogram("nFEDsWithMissingFEs","nFEDsWithMissingFEs",
                                       "Number of FEDs with missing FE unit payloads per event","");
}

//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDMonitorPlugin);
