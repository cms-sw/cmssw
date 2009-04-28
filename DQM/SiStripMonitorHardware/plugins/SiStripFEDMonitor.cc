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
// $Id: SiStripFEDMonitor.cc,v 1.11 2009/04/22 12:05:41 amagnan Exp $
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
  
  //FED errors
  FEDErrors lFedErrors;

  //initialise map of fedId/bad channel number
  std::map<unsigned int,std::pair<unsigned short,unsigned short> > badChannelFraction;
  std::pair<std::map<unsigned int,std::pair<unsigned short,unsigned short> >::iterator,bool> alreadyThere;

  //loop over siStrip FED IDs
  for (unsigned int fedId = FEDNumbering::MINSiStripFEDID; 
       fedId <= FEDNumbering::MAXSiStripFEDID; 
       fedId++) {
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);

    //create an object to fill all errors
    lFedErrors.initialise(fedId);
    FEDErrors::FEDLevelErrors & lFedLevelErrors = lFedErrors.getFEDLevelErrors();
    bool lFullDebug = false;
 
    //check data exists
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

    //Fill TkHistoMap:
    //1--Add all channels of a FED if anyFEDErrors or corruptBuffer
    if (lFedErrors.anyFEDErrors() || (lFedErrors.getFEDLevelErrors()).CorruptBuffer){
      for (unsigned int iCh = 0; 
	   iCh < sistrip::FEDCH_PER_FED; 
	   iCh++) {
	const FedChannelConnection & lConnection = cabling_->connection(fedId,iCh);
	if (!lConnection.isConnected()) continue;
	unsigned int detid = lConnection.detId();
	unsigned short nChInModule = lConnection.nApvPairs();
	alreadyThere = badChannelFraction.insert(std::pair<unsigned int,std::pair<unsigned short,unsigned short> >(detid,std::pair<unsigned short,unsigned short>(nChInModule,1)));
	if (!alreadyThere.second) ((alreadyThere.first)->second).second += 1;
      }

      lFedErrors.getFELevelErrors().clear();
      lFedErrors.getChannelLevelErrors().clear();

      assert(lFedErrors.getFELevelErrors().size() == 0);
      assert(lFedErrors.getChannelLevelErrors().size() == 0);
    }

    //if missing FEs or BadMajAddresses, fill channels vec with all channels from FE

    std::vector<FEDErrors::FELevelErrors> & lFeVec = lFedErrors.getFELevelErrors();
    unsigned int nBadFEs = lFeVec.size();
    std::vector<std::pair<unsigned int, bool> > & lBadChannels = lFedErrors.getBadChannels();

    //fill a map of affected FEs to not duplicate with badChannels
    std::map<unsigned short,bool> lFeMap;
    lFeMap.clear();

    for (unsigned int ife(0); ife<nBadFEs; ife++) {
      unsigned short feNumber = (lFeVec.at(ife)).FeID;
      lFeMap.insert(std::pair<unsigned short, bool>(feNumber,true));
      for (unsigned int feUnitCh = 0; feUnitCh < sistrip::FEDCH_PER_FEUNIT; feUnitCh++) {
	unsigned int iCh = feNumber*sistrip::FEDCH_PER_FEUNIT+feUnitCh;

	const FedChannelConnection & lConnection = cabling_->connection(fedId,iCh);
	if (!lConnection.isConnected()) continue;
	unsigned int detid = lConnection.detId();
	unsigned short nChInModule = lConnection.nApvPairs();
	alreadyThere = badChannelFraction.insert(std::pair<unsigned int,std::pair<unsigned short,unsigned short> >(detid,std::pair<unsigned short,unsigned short>(nChInModule,1)));
	if (!alreadyThere.second) ((alreadyThere.first)->second).second += 1;

      }
    }


    for (unsigned int iCh(0); iCh<lBadChannels.size(); iCh++) {
      if (lBadChannels.at(iCh).second) {
	unsigned short feNumber = static_cast<unsigned int>(iCh*1./sistrip::FEDCH_PER_FEUNIT);
	if (lFeMap.find(feNumber) != lFeMap.end()) continue;
	const FedChannelConnection & lConnection = cabling_->connection(fedId,lBadChannels.at(iCh).first);
	if (!lConnection.isConnected()) continue;
	unsigned int detid = lConnection.detId();
	unsigned short nChInModule = lConnection.nApvPairs();
	alreadyThere = badChannelFraction.insert(std::pair<unsigned int,std::pair<unsigned short,unsigned short> >(detid,std::pair<unsigned short,unsigned short>(nChInModule,1)));
	if (!alreadyThere.second) ((alreadyThere.first)->second).second += 1;

      }
    }
  }//loop over FED IDs
  
  fedHists_.fillCountersHistograms(FEDErrors::getFEDErrorsCounters());

  //match fedId/channel with detid

  std::map<unsigned int,std::pair<unsigned short,unsigned short> >::iterator fracIter;
  std::vector<std::pair<unsigned int,unsigned int> >::iterator chanIter;

  //std::cout << " --- Number of bad channels to fill in tkHistoMap = " << badChannelFraction.size() << std::endl;
  //int ele = 0;
  for (fracIter = badChannelFraction.begin(); fracIter!=badChannelFraction.end(); fracIter++){
    uint32_t detid = fracIter->first;
    //if ((fracIter->second).second != 0) {
    //std::cout << "------ ele #" << ele << ", Frac for detid #" << detid << " = " <<(fracIter->second).second << "/" << (fracIter->second).first << std::endl;
    //}
    unsigned short nTotCh = (fracIter->second).first;
    unsigned short nBadCh = (fracIter->second).second;
    assert (nTotCh >= nBadCh);
    if (nTotCh != 0) fedHists_.fillTkHistoMap(detid,static_cast<float>(nBadCh)/nTotCh);
    //ele++;
  }

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
    if (!printDebug_) return false;
  }
  //FE unit overflows
  if (!bufferBase->checkNoFEOverflows()) { 
    lFedLevelErrors.FEsOverflow = true;
    if (!printDebug_) return false;
  }
  
  //need to construct full object to go any further
  std::auto_ptr<const sistrip::FEDBuffer> buffer;
  buffer.reset(new sistrip::FEDBuffer(rawData.data(),rawData.size(),true));
  
  //payload checks
  if (!aFedErrors.anyFEDErrors()) {
    //corrupt buffer checks
    if (!buffer->doCorruptBufferChecks()) {
      lFedLevelErrors.CorruptBuffer = true;
      if (!printDebug_) return false;
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

  for (unsigned int iCh = 0; iCh < sistrip::FEDCH_PER_FED; iCh++) {
    bool isGood = true;
    if (!cabling_->connection(fedId,iCh).isConnected()) isGood = false;
    if (!buffer->feGood(static_cast<unsigned int>(iCh*1./sistrip::FEDCH_PER_FEUNIT))) isGood = false;

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

  }

  return !foundError;
}


//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDMonitorPlugin);
