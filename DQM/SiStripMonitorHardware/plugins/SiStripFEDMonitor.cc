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
// $Id: SiStripFEDMonitor.cc,v 1.10 2009/03/27 10:36:23 nc302 Exp $
//
//Modified        :  Anne-Marie Magnan
//   ---- 2009/04/21 : histogram management put in separate class
//                     struct helper to simplify arguments of functions
//

#include <sstream>
#include <memory>
#include <list>
#include <algorithm>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"

#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripMonitorHardware/interface/FEDHistograms.hh"


//
// Class declaration
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

  //return true if there were no errors at the level they are analysing
  //ie analyze FED returns true if there were no FED level errors which prevent the whole FED being unpacked
  bool analyzeFED(const FEDRawData& rawData, 
		  unsigned int fedId,
		  FEDHistograms::FEDCounters & aFEDLevelCounters
		  );

  bool analyzeFEUnits(const sistrip::FEDBuffer* buffer, 
		      unsigned int fedId,
		      FEDHistograms::FECounters & aFELevelCounters
                      );

  bool analyzeChannels(const sistrip::FEDBuffer* buffer, 
		       unsigned int fedId,
                       std::list<unsigned int>* badChannelList,
                       std::list<unsigned int>* activeBadChannelList
		       );
  

  //tag of FEDRawData collection
  edm::InputTag rawDataTag_;
  //histogram helper class
  FEDHistograms fedHists_;
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
  std::vector< std::vector<bool> > activeChannels_;
};


//
// Constructors and destructor
//

SiStripFEDMonitorPlugin::SiStripFEDMonitorPlugin(const edm::ParameterSet& iConfig)
  : rawDataTag_(iConfig.getUntrackedParameter<edm::InputTag>("RawDataTag",edm::InputTag("source",""))),
    folderName_(iConfig.getUntrackedParameter<std::string>("HistogramFolderName","SiStrip/ReadoutView/FedMonitoringSummary")),
    fillAllDetailedHistograms_(iConfig.getUntrackedParameter<bool>("FillAllDetailedHistograms",false)),
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
  
  fedHists_.initialise(iConfig,pDebugStream);


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
SiStripFEDMonitorPlugin::analyze(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{
  //update cabling
  updateCabling(iSetup);
  
  //get raw data
  edm::Handle<FEDRawDataCollection> rawDataCollectionHandle;
  iEvent.getByLabel(rawDataTag_,rawDataCollectionHandle);
  const FEDRawDataCollection& rawDataCollection = *rawDataCollectionHandle;
  
  //FED counters
  FEDHistograms::FEDCounters fedLevelCounters;
  fedLevelCounters.nFEDErrors = 0;
  fedLevelCounters.nDAQProblems = 0;
  fedLevelCounters.nFEDsWithFEProblems = 0;
  fedLevelCounters.nCorruptBuffers = 0;
  fedLevelCounters.nBadActiveChannels = 0;
  fedLevelCounters.nFEDsWithFEOverflows = 0;
  fedLevelCounters.nFEDsWithFEBadMajorityAddresses = 0;
  fedLevelCounters.nFEDsWithMissingFEs = 0;
  
  //loop over siStrip FED IDs
  for (unsigned int fedId = FEDNumbering::MINSiStripFEDID; 
       fedId <= FEDNumbering::MAXSiStripFEDID; 
       fedId++) {
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);
    //check data exists
    if (!fedData.size() || !fedData.data()) {
      bool fedHasCabledChannels = false;
      for (unsigned int iCh = 0; 
	   iCh < sistrip::FEDCH_PER_FED; 
	   iCh++) {
        if (cabling_->connection(fedId,iCh).isConnected()) {
          fedHasCabledChannels = true;
          break;
        }
      }
      if (fedHasCabledChannels) {
        fedHists_.fillHistogram("DataMissing",fedId);
        fedHists_.fillHistogram("AnyDAQProblems",fedId);
      }
      continue;
    } else {
      fedHists_.fillHistogram("DataPresent",fedId);
    }
    //check for problems and fill detailed histograms
    const bool anyFEDErrors = !analyzeFED(fedData,
					  fedId,
					  fedLevelCounters
					  );
    if (anyFEDErrors) fedHists_.fillHistogram("AnyFEDErrors",fedId);
  }//loop over FED IDs
  
  fedHists_.fillCountersHistogram(fedLevelCounters);

}

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripFEDMonitorPlugin::beginJob(const edm::EventSetup&)
{
  //get DQM store
  dqm_ = &(*edm::Service<DQMStore>());
  dqm_->setCurrentFolder(folderName_);
  
  //this propagates dqm_ to the histoclass, must be called !
  fedHists_.bookTopLevelHistograms(dqm_,folderName_);
  
  const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;

  if (fillAllDetailedHistograms_) fedHists_.bookAllFEDHistograms();
  //mark all channels as inactive until they have been 'locked' at least once
  activeChannels_.resize(siStripFedIdMax+1);
  for (unsigned int fedId = siStripFedIdMin; 
       fedId <= siStripFedIdMax; 
       fedId++) {
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

bool SiStripFEDMonitorPlugin::analyzeFED(const FEDRawData& rawData, 
					 unsigned int fedId,
					 FEDHistograms::FEDCounters & fedLevelCounters
					 )
{
  //try to construct the basic buffer object (do not check payload)
  //if this fails then count it as an invalid buffer and stop checks since we can't understand things like buffer ordering
  std::auto_ptr<const sistrip::FEDBufferBase> bufferBase;
  try {
    bufferBase.reset(new sistrip::FEDBufferBase(rawData.data(),rawData.size()));
  } catch (const cms::Exception& e) {
    fedHists_.fillHistogram("InvalidBuffers",fedId);
    fedHists_.fillHistogram("AnyDAQProblems",fedId);
    (fedLevelCounters.nDAQProblems)++;
    (fedLevelCounters.nFEDErrors)++;
    //don't check anything else if the buffer is invalid
    return false;
  }
  //CRC checks
  //if CRC fails then don't continue as if the buffer has been corrupted in DAQ then anything else could be invalid
  if (!bufferBase->checkNoSlinkCRCError()) {
    fedHists_.fillHistogram("BadFEDCRCs",fedId);
    fedHists_.fillHistogram("AnyDAQProblems",fedId);
    (fedLevelCounters.nDAQProblems)++;
    (fedLevelCounters.nFEDErrors)++;
    return false;
  } else if (!bufferBase->checkCRC()) {
    fedHists_.fillHistogram("BadDAQCRCs",fedId);
    fedHists_.fillHistogram("AnyDAQProblems",fedId);
    (fedLevelCounters.nDAQProblems)++;
    (fedLevelCounters.nFEDErrors)++;
    return false;
  }
  //next check that it is a SiStrip buffer
  //if not then stop checks
  if (!bufferBase->checkSourceIDs() || !bufferBase->checkNoUnexpectedSourceID()) {
    fedHists_.fillHistogram("BadIDs",fedId);
    fedHists_.fillHistogram("AnyDAQProblems",fedId);
    (fedLevelCounters.nDAQProblems)++;
    (fedLevelCounters.nFEDErrors)++;
    return false;
  } 
  //if so then do DAQ header/trailer checks
  //if these fail then buffer may be incomplete and checking contents doesn't make sense
  else if (!bufferBase->doDAQHeaderAndTrailerChecks()) {
    fedHists_.fillHistogram("BadDAQPacket",fedId);
    fedHists_.fillHistogram("AnyDAQProblems",fedId);
    (fedLevelCounters.nDAQProblems)++;
    (fedLevelCounters.nFEDErrors)++;
    return false;
  }
  
  bool foundError = false;
  //now do checks on header
  //check that tracker special header is consistent
  if ( !(bufferBase->checkBufferFormat() && bufferBase->checkHeaderType() && bufferBase->checkReadoutMode() && bufferBase->checkAPVEAddressValid()) ) {
    fedHists_.fillHistogram("InvalidBuffers",fedId);
    fedHists_.fillHistogram("AnyDAQProblems",fedId);
    (fedLevelCounters.nDAQProblems)++;
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

  FEDHistograms::FECounters feLevelCounters;
  feLevelCounters.nFEOverflows = 0;
  feLevelCounters.nFEBadMajorityAddresses = 0;
  feLevelCounters.nFEMissing = 0;

  bool feUnitsGood = analyzeFEUnits(buffer.get(),
				    fedId,
				    feLevelCounters);
  if (!feUnitsGood) foundError = true;
  if (feLevelCounters.nFEOverflows) (fedLevelCounters.nFEDsWithFEOverflows)++;
  if (feLevelCounters.nFEBadMajorityAddresses) (fedLevelCounters.nFEDsWithFEBadMajorityAddresses)++;
  if (feLevelCounters.nFEMissing) (fedLevelCounters.nFEDsWithMissingFEs)++;
  if (feLevelCounters.nFEMissing || 
      feLevelCounters.nFEBadMajorityAddresses || 
      feLevelCounters.nFEOverflows) (fedLevelCounters.nFEDsWithFEProblems)++;
  
  //payload checks
  std::list<unsigned int> badChannels;
  std::list<unsigned int> activeBadChannels;
  if (checkPayload) {
    //corrupt buffer checks
    if (!buffer->doCorruptBufferChecks()) {
      fedHists_.fillHistogram("CorruptBuffers",fedId);
      (fedLevelCounters.nCorruptBuffers)++;
      foundError = true;
    }
    //if there has been a FED error then count it then check channels
    if (foundError) (fedLevelCounters.nFEDErrors)++;
    //channel checks
    analyzeChannels(buffer.get(),fedId,&badChannels,&activeBadChannels);
    if (badChannels.size()) {
      fedHists_.fillHistogram("BadChannelStatusBits",fedId);
    }
    if (activeBadChannels.size()) {
      foundError = true;
      (fedLevelCounters.nBadActiveChannels) += activeBadChannels.size();
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

bool SiStripFEDMonitorPlugin::analyzeFEUnits(const sistrip::FEDBuffer* buffer, 
					     unsigned int fedId,
					     FEDHistograms::FECounters & aFELevelCounters
					     )
{
  bool foundOverflow = false;
  bool foundBadMajority = false;
  bool foundMissing = false;
  for (unsigned int iFE = 0; iFE < sistrip::FEUNITS_PER_FED; iFE++) {
    if (buffer->feOverflow(iFE)) {
      fedHists_.bookFEDHistograms(fedId);
      fedHists_.fillHistogram("FEOverflowsForFED",iFE,fedId);
      foundOverflow = true;
      (aFELevelCounters.nFEOverflows)++;
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
    if (!buffer->fePresent(iFE)) {
      if (hasCabledChannels) {
	fedHists_.bookFEDHistograms(fedId);
        fedHists_.fillHistogram("FEMissingForFED",iFE,fedId);
        foundMissing = true;
        (aFELevelCounters.nFEMissing)++;
      }
      continue;
    }
    if (buffer->majorityAddressErrorForFEUnit(iFE)) {
      fedHists_.bookFEDHistograms(fedId);
      fedHists_.fillHistogram("BadMajorityAddressesForFED",iFE,fedId);
      foundBadMajority = true;
      (aFELevelCounters.nFEBadMajorityAddresses)++;
    }
  }
  if (foundOverflow) {
    fedHists_.fillHistogram("FEOverflows",fedId);
  }
  if (foundMissing) {
    fedHists_.fillHistogram("FEMissing",fedId);
  }
  if (foundBadMajority) {
    fedHists_.fillHistogram("BadMajorityAddresses",fedId);
  }
  bool foundError = foundOverflow || foundBadMajority || foundMissing;
  if (foundError) {
    fedHists_.fillHistogram("AnyFEProblems",fedId);
  }
  return !foundError;
}

bool SiStripFEDMonitorPlugin::analyzeChannels(const sistrip::FEDBuffer* buffer, 
					      unsigned int fedId,
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
        fedHists_.bookFEDHistograms(fedId,debugHeader);
        fedHists_.fillHistogram("BadAPVStatusBitsForFED",iCh*2+iAPV,fedId);
        foundError = true;
        channelWasBad = true;
      }
    }
    //add channel to bad channel list
    if (channelWasBad && badChannelList) badChannelList->push_back(iCh);
    if (channelWasBad && activeChannels_[fedId][iCh] && activeBadChannelList) activeBadChannelList->push_back(iCh);
    //fill histogram for active channels
    if (channelWasBad && activeChannels_[fedId][iCh] && !filledBadChannel) {
      fedHists_.fillHistogram("BadActiveChannelStatusBits",fedId);
      filledBadChannel = true;
    }
  }
  if (debugHeader) {
    for (unsigned int iCh = 0; iCh < sistrip::FEDCH_PER_FED; iCh++) {
      for (unsigned int iAPV = 0; iAPV < 2; iAPV++) {
        if (debugHeader->apvError(iCh,iAPV)) {
          fedHists_.bookFEDHistograms(fedId,debugHeader);
          fedHists_.fillHistogram("APVErrorBitsForFED",iCh*2+iAPV,fedId);
        }
        if (debugHeader->apvAddressError(iCh,iAPV)) {
          fedHists_.bookFEDHistograms(fedId,debugHeader);
          fedHists_.fillHistogram("APVAddressErrorBitsForFED",iCh*2+iAPV,fedId);
        }
      }
      if (debugHeader->unlocked(iCh)) {
        fedHists_.bookFEDHistograms(fedId,debugHeader);
        fedHists_.fillHistogram("UnlockedBitsForFED",iCh,fedId);
      }
      if (debugHeader->outOfSync(iCh)) {
        fedHists_.bookFEDHistograms(fedId,debugHeader);
        fedHists_.fillHistogram("OOSBitsForFED",iCh,fedId);
      }
    }
  }
  return !foundError;
}


//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDMonitorPlugin);
