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

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiStripMonitorHardware/interface/FEDHistograms.hh"
#include "DQM/SiStripMonitorHardware/interface/FEDErrors.hh"

#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"


//
// Class declaration
//

class SiStripFEDMonitorPlugin : public edm::EDAnalyzer
{
 public:
  explicit SiStripFEDMonitorPlugin(const edm::ParameterSet&);
  ~SiStripFEDMonitorPlugin();
 private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
				    const edm::EventSetup& context) override;
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
				  const edm::EventSetup& context) override;

  //update the cabling if necessary
  void updateCabling(const edm::EventSetup& eventSetup);

  static bool pairComparison(const std::pair<unsigned int, unsigned int> & pair1,
			     const std::pair<unsigned int, unsigned int> & pair2);

  void getMajority(const std::vector<std::pair<unsigned int,unsigned int> > & aFeMajVec,
		   unsigned int & aMajorityCounter,
		   std::vector<unsigned int> & afedIds);

  //tag of FEDRawData collection
  edm::InputTag rawDataTag_;
  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;
  edm::EDGetTokenT<EventWithHistory> heToken_;
  
  //histogram helper class
  FEDHistograms fedHists_;
  //folder name for histograms in DQMStore
  std::string topFolderName_;
  std::string folderName_;
  //book detailed histograms even if they will be empty (for merging)
  bool fillAllDetailedHistograms_;
  //do histos vs time with time=event number. Default time = orbit number (s)
  bool fillWithEvtNum_;
  //print debug messages when problems are found: 1=error debug, 2=light debug, 3=full debug
  unsigned int printDebug_;
  //bool printDebug_;
  //write the DQMStore to a root file at the end of the job
  bool writeDQMStore_;
  std::string dqmStoreFileName_;
  //the DQMStore
  DQMStore* dqm_;
  //FED cabling
  uint32_t cablingCacheId_;
  const SiStripFedCabling* cabling_;

  //add parameter to save computing time if TkHistoMap/Median/FeMajCheck are not enabled
  bool doTkHistoMap_;
  bool doMedHists_;
  bool doFEMajorityCheck_;

  unsigned int nEvt_;

  //FED errors
  //need class member for lumi histograms
  FEDErrors fedErrors_;
  unsigned int maxFedBufferSize_;
};


//
// Constructors and destructor
//

SiStripFEDMonitorPlugin::SiStripFEDMonitorPlugin(const edm::ParameterSet& iConfig)
  : rawDataTag_(iConfig.getUntrackedParameter<edm::InputTag>("RawDataTag",edm::InputTag("source",""))),
    topFolderName_(iConfig.getUntrackedParameter<std::string>("TopFolderName","SiStrip")),
    fillAllDetailedHistograms_(iConfig.getUntrackedParameter<bool>("FillAllDetailedHistograms",false)),
    fillWithEvtNum_(iConfig.getUntrackedParameter<bool>("FillWithEventNumber",false)),
    printDebug_(iConfig.getUntrackedParameter<unsigned int>("PrintDebugMessages",1)),
    //printDebug_(iConfig.getUntrackedParameter<bool>("PrintDebugMessages",false)),
    writeDQMStore_(iConfig.getUntrackedParameter<bool>("WriteDQMStore",false)),
    dqmStoreFileName_(iConfig.getUntrackedParameter<std::string>("DQMStoreFileName","DQMStore.root")),
    dqm_(0),
    cablingCacheId_(0),
    maxFedBufferSize_(0)
{
  std::string subFolderName = iConfig.getUntrackedParameter<std::string>("HistogramFolderName","ReadoutView");
  folderName_ = topFolderName_ + "/" + subFolderName;
	      

  rawDataToken_ = consumes<FEDRawDataCollection>(rawDataTag_);
  heToken_      = consumes<EventWithHistory>(edm::InputTag("consecutiveHEs") );

  //print config to debug log
  std::ostringstream debugStream;
  if (printDebug_>1) {
    debugStream << "[SiStripFEDMonitorPlugin]Configuration for SiStripFEDMonitorPlugin: " << std::endl
                << "[SiStripFEDMonitorPlugin]\tRawDataTag: " << rawDataTag_ << std::endl
                << "[SiStripFEDMonitorPlugin]\tHistogramFolderName: " << folderName_ << std::endl
                << "[SiStripFEDMonitorPlugin]\tFillAllDetailedHistograms? " << (fillAllDetailedHistograms_ ? "yes" : "no") << std::endl
		<< "[SiStripFEDMonitorPlugin]\tFillWithEventNumber?" << (fillWithEvtNum_ ? "yes" : "no") << std::endl
                << "[SiStripFEDMonitorPlugin]\tPrintDebugMessages? " << (printDebug_ ? "yes" : "no") << std::endl
                << "[SiStripFEDMonitorPlugin]\tWriteDQMStore? " << (writeDQMStore_ ? "yes" : "no") << std::endl;
    if (writeDQMStore_) debugStream << "[SiStripFEDMonitorPlugin]\tDQMStoreFileName: " << dqmStoreFileName_ << std::endl;
  }
  
  //don;t generate debug mesages if debug is disabled
  std::ostringstream* pDebugStream = (printDebug_>1 ? &debugStream : NULL);
  
  fedHists_.initialise(iConfig,pDebugStream);

  doTkHistoMap_ = fedHists_.tkHistoMapEnabled();

  doMedHists_ = fedHists_.cmHistosEnabled();

  doFEMajorityCheck_ = fedHists_.feMajHistosEnabled();

  if (printDebug_) {
    LogTrace("SiStripMonitorHardware") << debugStream.str();

    //debugStream.str("");

    //debugStream << " -- Quelle est la difference entre un canard ? " << std::endl 
    //	<< " -- Reponse: c'est qu'il a les deux pattes de la meme longueur, surtout la gauche." << std::endl;

    //edm::LogError("SiStripMonitorHardware") << debugStream.str();
  }

  nEvt_ = 0;

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
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //update cabling
  updateCabling(iSetup);
  
  //get raw data
  edm::Handle<FEDRawDataCollection> rawDataCollectionHandle;
  iEvent.getByToken(rawDataToken_,rawDataCollectionHandle);
  const FEDRawDataCollection& rawDataCollection = *rawDataCollectionHandle;
  
  fedErrors_.initialiseEvent();

  //add the deltaBX value if the product exist

  edm::Handle<EventWithHistory> he;
  iEvent.getByToken(heToken_,he);

  if(he.isValid() && !he.failedToGet()) {
    fedErrors_.fillEventProperties(he->deltaBX());
  }

  //initialise map of fedId/bad channel number
  std::map<unsigned int,std::pair<unsigned short,unsigned short> > badChannelFraction;

  unsigned int lNFEDMonitoring = 0;
  unsigned int lNFEDUnpacker = 0;
  unsigned int lNChannelMonitoring = 0;
  unsigned int lNChannelUnpacker = 0;

  unsigned int lNTotBadFeds = 0;
  unsigned int lNTotBadChannels = 0;
  unsigned int lNTotBadActiveChannels = 0;

  std::vector<std::vector<std::pair<unsigned int,unsigned int> > > lFeMajFrac;
  const unsigned int nParts = 4;
  if (doFEMajorityCheck_){
    lFeMajFrac.resize(nParts);
    //max nFE per partition
    lFeMajFrac[0].reserve(912);
    lFeMajFrac[1].reserve(1080);
    lFeMajFrac[2].reserve(768);
    lFeMajFrac[3].reserve(760);
  }

  maxFedBufferSize_ = 0;

  //loop over siStrip FED IDs
  for (unsigned int fedId = FEDNumbering::MINSiStripFEDID; 
       fedId <= FEDNumbering::MAXSiStripFEDID; 
       fedId++) {//loop over FED IDs
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);

    //create an object to fill all errors
    fedErrors_.initialiseFED(fedId,cabling_,tTopo);
    bool lFullDebug = false;
 
    //Do detailed check
    //first check if data exists
    bool lDataExist = fedErrors_.checkDataPresent(fedData);
    if (!lDataExist) {
      fedHists_.fillFEDHistograms(fedErrors_,0,lFullDebug);
      continue;
    }


 
    //check for problems and fill detailed histograms
    fedErrors_.fillFEDErrors(fedData,
			     lFullDebug,
			     printDebug_,
			     lNChannelMonitoring,
			     lNChannelUnpacker,
			     doMedHists_,
			     fedHists_.cmHistPointer(false),
			     fedHists_.cmHistPointer(true),
			     doFEMajorityCheck_,
			     lFeMajFrac
			     );

    //check filled in previous method.
    bool lFailUnpackerFEDcheck = fedErrors_.failUnpackerFEDCheck();

    fedErrors_.incrementFEDCounters();
    unsigned int lSize = fedData.size();
    if (lSize > maxFedBufferSize_){
      maxFedBufferSize_ = lSize;
    }
    //std::cout << " -- " << fedId << " " << lSize << std::endl;

    fedHists_.fillFEDHistograms(fedErrors_,lSize,lFullDebug);

    bool lFailMonitoringFEDcheck = fedErrors_.failMonitoringFEDCheck();
    if (lFailMonitoringFEDcheck) lNTotBadFeds++;


    //sanity check: if something changed in the unpacking code 
    //but wasn't propagated here
    //print only the summary, and more info if printDebug>1
    if (lFailMonitoringFEDcheck != lFailUnpackerFEDcheck) {
      if (printDebug_>1) {
	std::ostringstream debugStream;
	debugStream << " --- WARNING: FED " << fedId << std::endl 
		    << " ------ Monitoring FED check " ;
	if (lFailMonitoringFEDcheck) debugStream << "failed." << std::endl;
	else debugStream << "passed." << std::endl ;
	debugStream << " ------ Unpacker FED check " ;
	if (lFailUnpackerFEDcheck) debugStream << "failed." << std::endl;
	else debugStream << "passed." << std::endl ;
	edm::LogError("SiStripMonitorHardware") << debugStream.str();
      }

      if (lFailMonitoringFEDcheck) lNFEDMonitoring++;
      else if (lFailUnpackerFEDcheck) lNFEDUnpacker++;
    }


    //Fill TkHistoMap:
    //add an entry for all channels (good = 0), 
    //so that tkHistoMap knows which channels should be there.
    if (doTkHistoMap_ && !fedHists_.tkHistoMapPointer()) {
      edm::LogWarning("SiStripMonitorHardware") << " -- Fedid " << fedId
						<< ", TkHistoMap enabled but pointer is null." << std::endl;
    }

    fedErrors_.fillBadChannelList(doTkHistoMap_,
				  fedHists_.tkHistoMapPointer(),
				  fedHists_.getFedvsAPVpointer(),
				  lNTotBadChannels,
				  lNTotBadActiveChannels);
  }//loop over FED IDs


  if (doFEMajorityCheck_){
    for (unsigned int iP(0); iP<nParts; ++iP){
      //std::cout << " -- Partition " << iP << std::endl;
      //std::cout << " --- Number of elements in vec = " << lFeMajFrac[iP].size() << std::endl;
      if (lFeMajFrac[iP].size()==0) continue;
      std::sort(lFeMajFrac[iP].begin(),lFeMajFrac[iP].end(),SiStripFEDMonitorPlugin::pairComparison);

      unsigned int lMajorityCounter = 0;
      std::vector<unsigned int> lfedIds;

      getMajority(lFeMajFrac[iP],lMajorityCounter,lfedIds);
      //std::cout << " -- Found " << lfedIds.size() << " unique elements not matching the majority." << std::endl;
      fedHists_.fillMajorityHistograms(iP,static_cast<float>(lMajorityCounter)/lFeMajFrac[iP].size(),lfedIds);
    }
  }

  if ((lNTotBadFeds> 0 || lNTotBadChannels>0) && printDebug_>1) {
    std::ostringstream debugStream;
    debugStream << "[SiStripFEDMonitorPlugin] --- Total number of bad feds = " 
		<< lNTotBadFeds << std::endl
		<< "[SiStripFEDMonitorPlugin] --- Total number of bad channels = " 
		<< lNTotBadChannels << std::endl
		<< "[SiStripFEDMonitorPlugin] --- Total number of bad active channels = " 
		<< lNTotBadActiveChannels << std::endl;
    edm::LogInfo("SiStripMonitorHardware") << debugStream.str();
  }

  if ((lNFEDMonitoring > 0 || lNFEDUnpacker > 0 || lNChannelMonitoring > 0 || lNChannelUnpacker > 0) && printDebug_) {
    std::ostringstream debugStream;
    debugStream
      << "[SiStripFEDMonitorPlugin]-------------------------------------------------------------------------" << std::endl 
      << "[SiStripFEDMonitorPlugin]-------------------------------------------------------------------------" << std::endl 
      << "[SiStripFEDMonitorPlugin]-- Summary of differences between unpacker and monitoring at FED level : " << std::endl 
      << "[SiStripFEDMonitorPlugin] ---- Number of times monitoring fails but not unpacking = " << lNFEDMonitoring << std::endl 
      << "[SiStripFEDMonitorPlugin] ---- Number of times unpacking fails but not monitoring = " << lNFEDUnpacker << std::endl
      << "[SiStripFEDMonitorPlugin]-------------------------------------------------------------------------" << std::endl 
      << "[SiStripFEDMonitorPlugin]-- Summary of differences between unpacker and monitoring at Channel level : " << std::endl 
      << "[SiStripFEDMonitorPlugin] ---- Number of times monitoring fails but not unpacking = " << lNChannelMonitoring << std::endl 
      << "[SiStripFEDMonitorPlugin] ---- Number of times unpacking fails but not monitoring = " << lNChannelUnpacker << std::endl
      << "[SiStripFEDMonitorPlugin]-------------------------------------------------------------------------" << std::endl 
      << "[SiStripFEDMonitorPlugin]-------------------------------------------------------------------------" << std::endl ;
    edm::LogError("SiStripMonitorHardware") << debugStream.str();

  }

  FEDErrors::getFEDErrorsCounters().nTotalBadChannels = lNTotBadChannels;
  FEDErrors::getFEDErrorsCounters().nTotalBadActiveChannels = lNTotBadActiveChannels;

  //fedHists_.fillCountersHistograms(FEDErrors::getFEDErrorsCounters(), nEvt_);
  //time in seconds since beginning of the run or event number
  if (fillWithEvtNum_) fedHists_.fillCountersHistograms(FEDErrors::getFEDErrorsCounters(),FEDErrors::getChannelErrorsCounters(),maxFedBufferSize_,iEvent.id().event());
  else fedHists_.fillCountersHistograms(FEDErrors::getFEDErrorsCounters(),FEDErrors::getChannelErrorsCounters(),maxFedBufferSize_,iEvent.orbitNumber()/11223.);

  nEvt_++;

}//analyze method


bool SiStripFEDMonitorPlugin::pairComparison(const std::pair<unsigned int, unsigned int> & pair1,
					     const std::pair<unsigned int, unsigned int> & pair2){
  return (pair1.second < pair2.second) ;
}

void SiStripFEDMonitorPlugin::getMajority(const std::vector<std::pair<unsigned int,unsigned int> > & aFeMajVec,
					  unsigned int & aMajorityCounter,
					  std::vector<unsigned int> & afedIds) {
  
  
  unsigned int lMajAddress = 0;
  std::vector<std::pair<unsigned int,unsigned int> >::const_iterator lIter = aFeMajVec.begin();
  unsigned int lMajAddr = (*lIter).second;
  unsigned int lCounter = 0;
  
  //std::cout << " --- First element: addr = " << lMajAddr << " counter = " << lCounter << std::endl;
  unsigned int iele=0;
  //bool foundMaj = false;
  for ( ; lIter != aFeMajVec.end(); ++lIter,++iele) {
    //std::cout << " ---- Ele " << iele << " " << (*lIter).first << " " << (*lIter).second << " ref " << lMajAddr << std::endl;
    if ((*lIter).second == lMajAddr) {
      ++lCounter;
      //std::cout << " ----- =ref: Counter = " << lCounter << std::endl;
    }
    else {
      //std::cout << " ----- !=ref: Counter = " << lCounter << " Majority = " << aMajorityCounter << std::endl;
      if (lCounter > aMajorityCounter) {
	//std::cout << " ------ >Majority: " << std::endl;
	aMajorityCounter = lCounter;
	// AV bug here??
	lMajAddress = lMajAddr;
	//	lMajAddress = (*lIter).second;
	//foundMaj=true;
      }
      lCounter = 0;
      lMajAddr = (*lIter).second;
      --lIter;
      --iele;
    }
  }
  // AV Bug here? The check has to be done regardless foundMaj == false or true 
  //  if (!foundMaj) {
    if (lCounter > aMajorityCounter) {
      //std::cout << " ------ >Majority: " << std::endl;
      aMajorityCounter = lCounter;
      lMajAddress = lMajAddr;
    }
    //  }
  //std::cout << " -- found majority value for " << aMajorityCounter << " elements out of " << aFeMajVec.size() << "." << std::endl;
  //get list of feds with address different from majority in partition:      
  lIter = aFeMajVec.begin();
  afedIds.reserve(135);
  for ( ; lIter != aFeMajVec.end(); ++lIter ) {
    if((*lIter).second != lMajAddress) {
      afedIds.push_back((*lIter).first);
    }
    else {
      lIter += aMajorityCounter-1;
      if(lIter >= aFeMajVec.end()) {
	std::cout << "Here it is a bug: " << aMajorityCounter << " " << aFeMajVec.size() << " " << lIter - aFeMajVec.end() << std::endl;
      }
    }
  }
  //std::cout << " -- Found " << lfedIds.size() << " elements not matching the majority." << std::endl;
  if (afedIds.size()>0) {
    std::sort(afedIds.begin(),afedIds.end());
    std::vector<unsigned int>::iterator lIt = std::unique(afedIds.begin(),afedIds.end());
    afedIds.erase(lIt,afedIds.end());
  }

}

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripFEDMonitorPlugin::beginJob()
{
  //get DQM store
  dqm_ = &(*edm::Service<DQMStore>());
  dqm_->setCurrentFolder(folderName_);
  
  //this propagates dqm_ to the histoclass, must be called !
  fedHists_.bookTopLevelHistograms(dqm_);
  
  if (fillAllDetailedHistograms_) fedHists_.bookAllFEDHistograms();

  nEvt_ = 0;

  //const unsigned int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  //const unsigned int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;

  //mark all channels as inactive until they have been 'locked' at least once
  //   activeChannels_.resize(siStripFedIdMax+1);
  //   for (unsigned int fedId = siStripFedIdMin; 
  //        fedId <= siStripFedIdMax; 
  //        fedId++) {
  //     activeChannels_[fedId].resize(sistrip::FEDCH_PER_FED,false);
  //   }
  



}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripFEDMonitorPlugin::endJob()
{
  if (writeDQMStore_) dqm_->save(dqmStoreFileName_);
}



void 
SiStripFEDMonitorPlugin::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					      const edm::EventSetup& context)
{

  fedErrors_.initialiseLumiBlock();

}


void 
SiStripFEDMonitorPlugin::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					    const edm::EventSetup& context)
{
  fedHists_.fillLumiHistograms(fedErrors_.getLumiErrors());
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


//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDMonitorPlugin);
