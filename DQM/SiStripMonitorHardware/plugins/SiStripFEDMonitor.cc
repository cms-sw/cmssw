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
// $Id: SiStripFEDMonitor.cc,v 1.14 2009/05/08 16:12:21 amagnan Exp $
//
//Modified        :  Anne-Marie Magnan
//   ---- 2009/04/21 : histogram management put in separate class
//                     struct helper to simplify arguments of functions
//   ---- 2009/04/22 : add TkHistoMap with % of bad channels per module
//   ---- 2009/04/27 : create FEDErrors class 

#include <sstream>
#include <memory>
#include <list>
#include <algorithm>
#include <cassert>

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
#include "DQM/SiStripMonitorHardware/interface/FEDErrors.hh"


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

  //perform a sanity check with unpacking code check
  bool failUnpackerFEDCheck(const FEDRawData& rawData);

  //return true if there were no errors at the level they are analysing
  //ie analyze FED returns true if there were no FED level errors which prevent the whole FED being unpacked
  bool analyzeFED(const FEDRawData& rawData, 
		  unsigned int fedId,
		  FEDErrors & aFEDError,
		  bool & aFullDebug
		  );

  bool analyzeFEUnits(const sistrip::FEDBuffer* buffer, 
		      unsigned int fedId,
		      FEDErrors & aFEDError
		      );

  bool analyzeChannels(const sistrip::FEDBuffer* buffer, 
		       unsigned int fedId,
                       FEDErrors & aFEDError,
		       bool & aFullDebug
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

  //add parameter to save computing time if TkHistoMap are not filled
  bool doTkHistoMap_;

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

  doTkHistoMap_ = fedHists_.isTkHistoMapEnabled();


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
  
  static bool lFirstEvent = true;

  //get raw data
  edm::Handle<FEDRawDataCollection> rawDataCollectionHandle;
  iEvent.getByLabel(rawDataTag_,rawDataCollectionHandle);
  const FEDRawDataCollection& rawDataCollection = *rawDataCollectionHandle;
  
  //FED errors
  FEDErrors lFedErrors;

  //initialise map of fedId/bad channel number
  std::map<unsigned int,std::pair<unsigned short,unsigned short> > badChannelFraction;
  std::pair<std::map<unsigned int,std::pair<unsigned short,unsigned short> >::iterator,bool> alreadyThere;

  unsigned int lNMonitoring = 0;
  unsigned int lNUnpacker = 0;

  unsigned int lNTotBadFeds = 0;
  unsigned int lNTotBadChannels = 0;

  //loop over siStrip FED IDs
  for (unsigned int fedId = FEDNumbering::MINSiStripFEDID; 
       fedId <= FEDNumbering::MAXSiStripFEDID; 
       fedId++) {//loop over FED IDs
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);

    //create an object to fill all errors
    lFedErrors.initialise(fedId);
    FEDErrors::FEDLevelErrors & lFedLevelErrors = lFedErrors.getFEDLevelErrors();
    bool lFullDebug = false;
 

    //Do detailed check
    //first check if data exists
    if (!fedData.size() || !fedData.data()) {
      for (unsigned int iCh = 0; 
	   iCh < sistrip::FEDCH_PER_FED; 
	   iCh++) {
        if (cabling_->connection(fedId,iCh).isConnected()) {
          lFedLevelErrors.HasCabledChannels = true;
	  lFedLevelErrors.DataMissing = true;
          break;
        }
      }
      //if no data, fill histos and go to next FED
      fedHists_.fillFEDHistograms(lFedErrors,lFullDebug);
      continue;
    } else {
      lFedLevelErrors.DataPresent = true;
    }

    //check for problems and fill detailed histograms
    analyzeFED(fedData,
	       fedId,
	       lFedErrors,
	       lFullDebug
	       );

    lFedErrors.incrementFEDCounters();
    fedHists_.fillFEDHistograms(lFedErrors,lFullDebug);

    bool lFailMonitoringFEDcheck = lFedErrors.anyFEDErrors() || (lFedErrors.getFEDLevelErrors()).CorruptBuffer;

    if (lFailMonitoringFEDcheck) lNTotBadFeds++;

    //Do exactly same check as unpacker
    bool lFailUnpackerFEDcheck = failUnpackerFEDCheck(fedData);
    
    //sanity check: if something changed in the unpacking code 
    //but wasn't propagated here
    if (lFailMonitoringFEDcheck != lFailUnpackerFEDcheck) {
      std::cout << " --- WARNING: FED " << fedId << std::endl 
	       << " ------ Monitoring FED check " ;
      if (lFailMonitoringFEDcheck) std::cout << "failed." << std::endl;
      else std::cout << "passed." << std::endl ;
      std::cout << " ------ Unpacker FED check " ;
      if (lFailUnpackerFEDcheck) std::cout << "failed." << std::endl;
      else std::cout << "passed." << std::endl ;

      if (lFailMonitoringFEDcheck) lNMonitoring++;
      else if (lFailUnpackerFEDcheck) lNUnpacker++;

    }


    if (doTkHistoMap_){//if TkHistMap is enabled

      //Fill TkHistoMap:
      //1--Add all channels of a FED if anyFEDErrors or corruptBuffer
      //2--Add all channels anyway with 0 if no errors, so TkHistoMap is filled for all valid channels ...

      std::vector<FEDErrors::FELevelErrors> & lFeVec = lFedErrors.getFELevelErrors();
      unsigned int nBadFEs = lFeVec.size();
      std::vector<std::pair<unsigned int, bool> > & lBadChannels = lFedErrors.getBadChannels();

      //unsigned int nBadChans = 0;
      for (unsigned int iCh = 0; 
	   iCh < sistrip::FEDCH_PER_FED; 
	   iCh++) {//loop on channels
	const FedChannelConnection & lConnection = cabling_->connection(fedId,iCh);
	if (!lConnection.isConnected()) continue;
	
	unsigned int feNumber = static_cast<unsigned int>(iCh/sistrip::FEDCH_PER_FEUNIT);
	
	bool isBadFE = false;
	for (unsigned int badfe(0); badfe<nBadFEs; badfe++) {
	  if ((lFeVec.at(badfe)).FeID == feNumber) {
	    isBadFE = true;
	    break;
	  }
	}

	bool isBadChan = false;
	for (unsigned int badCh(0); badCh<lBadChannels.size(); badCh++) {
	  if (lBadChannels.at(badCh).first == iCh) {
	    isBadChan = true;
	    break;
	  }
	}

	unsigned int detid = lConnection.detId();
	if (!detid || detid == sistrip::invalid32_) continue;
	unsigned short nChInModule = lConnection.nApvPairs();

	if (lFailMonitoringFEDcheck || isBadFE || isBadChan) {
	  alreadyThere = badChannelFraction.insert(std::pair<unsigned int,std::pair<unsigned short,unsigned short> >(detid,std::pair<unsigned short,unsigned short>(nChInModule,1)));
	  if (!alreadyThere.second) ((alreadyThere.first)->second).second += 1;
	  //nBadChans++;
	  lNTotBadChannels++;
	}
	else {
	  alreadyThere = badChannelFraction.insert(std::pair<unsigned int,std::pair<unsigned short,unsigned short> >(detid,std::pair<unsigned short,unsigned short>(nChInModule,0)));
	}

      }//loop on channels

      //if (nBadChans>0) std::cout << "-------- FED " << fedId << ", " << nBadChans << " bad channels." << std::endl;

    }//if TkHistMap is enabled
  }//loop over FED IDs
  
  if (lNTotBadFeds> 0 || lNTotBadChannels>0 ) {
    std::cout << " --- Total number of bad feds = " 
	      << lNTotBadFeds << std::endl
	      << " --- Total number of bad channels = " 
	      << lNTotBadChannels << std::endl;
  }

  if (lNMonitoring > 0 || lNUnpacker > 0) {
    std::cout
      << "-------------------------------------------------------------------------" << std::endl 
      << "-------------------------------------------------------------------------" << std::endl 
      << "-- Summary of differences between unpacker and monitoring at FED level : " << std::endl 
      << " ---- Number of times monitoring fails but not unpacking = " << lNMonitoring << std::endl 
      << " ---- Number of times unpacking fails but not monitoring = " << lNUnpacker << std::endl
      << "-------------------------------------------------------------------------" << std::endl 
      << "-------------------------------------------------------------------------" << std::endl ;
  }

  fedHists_.fillCountersHistograms(FEDErrors::getFEDErrorsCounters());

  //match fedId/channel with detid

  if (doTkHistoMap_) {//if TkHistoMap is enabled
    std::map<unsigned int,std::pair<unsigned short,unsigned short> >::iterator fracIter;
    std::vector<std::pair<unsigned int,unsigned int> >::iterator chanIter;

    //int ele = 0;
    //int nBadChannels = 0;
    for (fracIter = badChannelFraction.begin(); fracIter!=badChannelFraction.end(); fracIter++){
      uint32_t detid = fracIter->first;
      //if ((fracIter->second).second != 0) {
      //std::cout << "------ ele #" << ele << ", Frac for detid #" << detid << " = " <<(fracIter->second).second << "/" << (fracIter->second).first << std::endl;
      //nBadChannels++;
      //}
      unsigned short nTotCh = (fracIter->second).first;
      unsigned short nBadCh = (fracIter->second).second;
      assert (nTotCh >= nBadCh);
      if (nTotCh != 0) fedHists_.fillTkHistoMap(detid,static_cast<float>(nBadCh)/nTotCh);
      //ele++;
    }
    //std::cout << "--- Total number of badChannels in map = " << nBadChannels << std::endl;

  }//if TkHistoMap is enabled

  lFirstEvent = false;

}//analyze method

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripFEDMonitorPlugin::beginJob(const edm::EventSetup&)
{
  //get DQM store
  dqm_ = &(*edm::Service<DQMStore>());
  dqm_->setCurrentFolder(folderName_);
  
  //this propagates dqm_ to the histoclass, must be called !
  fedHists_.bookTopLevelHistograms(dqm_);
  
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

bool SiStripFEDMonitorPlugin::failUnpackerFEDCheck(const FEDRawData& rawData)
{
  // construct FEDBuffer
  std::auto_ptr<sistrip::FEDBuffer> buffer;
  try {
    buffer.reset(new sistrip::FEDBuffer(rawData.data(),rawData.size()));
    if (!buffer->doChecks()) return true;
    //throw cms::Exception("FEDBuffer") << "FED Buffer check fails.";
  }
  catch (const cms::Exception& e) { 
    if ( edm::isDebugEnabled() ) {
      edm::LogWarning("DQM:SiStripFEDMonitor") << e.what();
    }
    return true;
  }
  return false;

}

bool SiStripFEDMonitorPlugin::analyzeFED(const FEDRawData& rawData, 
					 unsigned int fedId,
					 FEDErrors & aFedErrors,
					 bool & aFullDebug
					 )
{
  //try to construct the basic buffer object (do not check payload)
  //if this fails then count it as an invalid buffer and stop checks since we can't understand things like buffer ordering

  FEDErrors::FEDLevelErrors & lFedLevelErrors = aFedErrors.getFEDLevelErrors();

  std::auto_ptr<const sistrip::FEDBufferBase> bufferBase;
  try {
    bufferBase.reset(new sistrip::FEDBufferBase(rawData.data(),rawData.size()));
  } catch (const cms::Exception& e) {
    lFedLevelErrors.InvalidBuffers = true;
    //don't check anything else if the buffer is invalid
    return false;
  }
  //CRC checks
  //if CRC fails then don't continue as if the buffer has been corrupted in DAQ then anything else could be invalid
  if (!bufferBase->checkNoSlinkCRCError()) {
    lFedLevelErrors.BadFEDCRCs = true;
    return false;
  } else if (!bufferBase->checkCRC()) {
    lFedLevelErrors.BadDAQCRCs = true;
    return false;
  }
  //next check that it is a SiStrip buffer
  //if not then stop checks
  if (!bufferBase->checkSourceIDs() || !bufferBase->checkNoUnexpectedSourceID()) {
    lFedLevelErrors.BadIDs = true;
    return false;
  } 
  //if so then do DAQ header/trailer checks
  //if these fail then buffer may be incomplete and checking contents doesn't make sense
  else if (!bufferBase->doDAQHeaderAndTrailerChecks()) {
    lFedLevelErrors.BadDAQPacket = true;
    return false;
  }

  //now do checks on header
  //check that tracker special header is consistent
  if ( !(bufferBase->checkBufferFormat() && bufferBase->checkHeaderType() && bufferBase->checkReadoutMode() && bufferBase->checkAPVEAddressValid()) ) {
    lFedLevelErrors.InvalidBuffers = true;
    //keep running only in debug mode....
    //if (!printDebug_) return false;
  }
  //FE unit overflows
  if (!bufferBase->checkNoFEOverflows()) { 
    lFedLevelErrors.FEsOverflow = true;
    //if (!printDebug_) return false;
  }
  
  //need to construct full object to go any further
  std::auto_ptr<const sistrip::FEDBuffer> buffer;
  buffer.reset(new sistrip::FEDBuffer(rawData.data(),rawData.size(),true));
  static bool lFirst = true;
  if (lFirst) std::cout << "--- FEDID " << fedId << ", readout mode = " << buffer->readoutMode() << std::endl;
  lFirst = false;

  //payload checks
  if (!aFedErrors.anyFEDErrors()) {
    //corrupt buffer checks
    if (!buffer->doCorruptBufferChecks()) {
      lFedLevelErrors.CorruptBuffer = true;
      //if (!printDebug_) return false;
    }

    //fe check... 
    analyzeFEUnits(buffer.get(),
		   fedId,
		   aFedErrors
		   );
    
    //channel checks
    analyzeChannels(buffer.get(),
		    fedId,
		    aFedErrors,
		    aFullDebug
		    );

  }
  
  if (aFedErrors.printDebug() && printDebug_) {
    const sistrip::FEDBufferBase* debugBuffer = NULL;
    if (buffer.get()) debugBuffer = buffer.get();
    else if (bufferBase.get()) debugBuffer = bufferBase.get();
    if (debugBuffer) {
      std::vector<FEDErrors::APVLevelErrors> & lChVec = aFedErrors.getAPVLevelErrors();
      std::ostringstream debugStream;
      if (lChVec.size()) {
	std::sort(lChVec.begin(),lChVec.end());
        debugStream << "Cabled channels which had errors: ";
	
        for (unsigned int iBadCh(0); iBadCh < lChVec.size(); iBadCh++) {
          aFedErrors.print(lChVec.at(iBadCh),debugStream);
        }
        debugStream << std::endl;
        debugStream << "Active (have been unlocked in at least one event) cabled channels which had errors: ";
	for (unsigned int iBadCh(0); iBadCh < lChVec.size(); iBadCh++) {
          if ((lChVec.at(iBadCh)).IsActive) aFedErrors.print(lChVec.at(iBadCh),debugStream);
        }

      }
      debugStream << (*debugBuffer) << std::endl;
      debugBuffer->dump(debugStream);
      debugStream << std::endl;
      edm::LogInfo("SiStripMonitorHardware") << "Errors found in FED " << fedId;
      edm::LogVerbatim("SiStripMonitorHardware") << debugStream.str();
    }
  }
  
  return !(aFedErrors.anyFEDErrors());
}

bool SiStripFEDMonitorPlugin::analyzeFEUnits(const sistrip::FEDBuffer* buffer, 
					     unsigned int fedId,
					     FEDErrors & aFedErrors
					     )
{
  bool foundOverflow = false;
  bool foundBadMajority = false;
  bool foundMissing = false;
  for (unsigned int iFE = 0; iFE < sistrip::FEUNITS_PER_FED; iFE++) {
    
    FEDErrors::FELevelErrors lFeErr;
    lFeErr.FeID = iFE;
    lFeErr.Overflow = false;
    lFeErr.Missing = false;
    lFeErr.BadMajorityAddress = false;

    if (buffer->feOverflow(iFE)) {
      lFeErr.Overflow = true;
      foundOverflow = true;
      aFedErrors.addBadFE(lFeErr);
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
	lFeErr.Missing = true;
        foundMissing = true;
	aFedErrors.addBadFE(lFeErr);
      }
      continue;
    }
    if (buffer->majorityAddressErrorForFEUnit(iFE)) {
      foundBadMajority = true;
      aFedErrors.addBadFE(lFeErr);
    }
  }

  return !(foundOverflow || foundMissing || foundBadMajority);

}

bool SiStripFEDMonitorPlugin::analyzeChannels(const sistrip::FEDBuffer* buffer, 
					      unsigned int fedId,
					      FEDErrors & aFedErrors,
					      bool & aFullDebug
 					      )
{
  bool foundError = false;

  const sistrip::FEDFEHeader* header = buffer->feHeader();
  const sistrip::FEDFullDebugHeader* debugHeader = dynamic_cast<const sistrip::FEDFullDebugHeader*>(header);

  aFullDebug = debugHeader;

  for (unsigned int iCh = 0; iCh < sistrip::FEDCH_PER_FED; iCh++) {//loop on channels
    bool isGood = true;

    const FedChannelConnection & lConnection = cabling_->connection(fedId,iCh);

    bool lFailUnpackerChannelCheck = !buffer->channelGood(lConnection.fedCh()) && lConnection.isConnected();
    bool lFailMonitoringChannelCheck = false;

    if (!lConnection.isConnected()) isGood = false;
    else if (!buffer->feGood(static_cast<unsigned int>(iCh/sistrip::FEDCH_PER_FEUNIT))) {
      isGood = false;
      lFailMonitoringChannelCheck = true;
    }    

    if (debugHeader) {
      if (!debugHeader->unlocked(iCh)) activeChannels_[fedId][iCh] = true;
    } else {
      if (header->checkChannelStatusBits(iCh)) activeChannels_[fedId][iCh] = true;
    }

    FEDErrors::ChannelLevelErrors lChErr;
    lChErr.ChannelID = iCh;
    lChErr.IsActive = activeChannels_[fedId][iCh];
    lChErr.Unlocked = false;
    lChErr.OutOfSync = false;


    bool lFirst = true;

    for (unsigned int iAPV = 0; iAPV < 2; iAPV++) {//loop on APVs

      FEDErrors::APVLevelErrors lAPVErr;
      lAPVErr.APVID = 2*iCh+iAPV;
      lAPVErr.ChannelID = iCh;
      lAPVErr.IsActive = activeChannels_[fedId][iCh];
      lAPVErr.APVStatusBit = false;
      lAPVErr.APVError = false;
      lAPVErr.APVAddressError = false;

      if (!header->checkStatusBits(iCh,iAPV) && isGood) {
	lFailMonitoringChannelCheck = true;
	lAPVErr.APVStatusBit = true;
	foundError = true;
      }

      if (debugHeader) {
	if (debugHeader->apvError(iCh,iAPV)) {
	  lAPVErr.APVError = true;
	}
        if (debugHeader->apvAddressError(iCh,iAPV)) {
          lAPVErr.APVAddressError = true;
        }
      }

      if ( (lAPVErr.APVStatusBit && isGood) || 
	   lAPVErr.APVError || 
	   lAPVErr.APVAddressError
	   ) aFedErrors.addBadAPV(lAPVErr, lFirst);
    }//loop on APVs

    if (debugHeader) {
      if (debugHeader->unlocked(iCh)) {
	lChErr.Unlocked = true;
      }
      if (debugHeader->outOfSync(iCh)) {
	lChErr.OutOfSync = true;
      }
      if (lChErr.Unlocked || lChErr.OutOfSync) aFedErrors.addBadChannel(lChErr);
    }


//     if (lFailUnpackerChannelCheck != lFailMonitoringChannelCheck) {
      
//       std::cout << " ------ WARNING: FED " << fedId << ", channel " << iCh << std::endl 
// 	       << " --------- Monitoring Channel check " ;
//       if (lFailMonitoringChannelCheck) std::cout << "failed." << std::endl;
//       else std::cout << "passed." << std::endl ;
//       std::cout << " --------- Unpacker Channel check " ;
//       if (lFailUnpackerChannelCheck) std::cout << "failed." << std::endl;
//       else std::cout << "passed." << std::endl;

      
//     }

  }//loop on channels

  return !foundError;
}


//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDMonitorPlugin);
