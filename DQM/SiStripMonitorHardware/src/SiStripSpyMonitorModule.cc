// Original Author:  Anne-Marie Magnan
//         Created:  2010/01/11
// $Id: SiStripSpyMonitorModule.cc,v 1.1 2012/10/15 09:02:48 threus Exp $
//

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
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

// For plotting.
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripMonitorHardware/interface/SiStripFEDSpyBuffer.h"
#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUtilities.h"
#include "DQM/SiStripMonitorHardware/interface/SPYHistograms.h"

//
// Class declaration
//

class SiStripSpyMonitorModule : public edm::EDAnalyzer
{
 public:


  explicit SiStripSpyMonitorModule(const edm::ParameterSet&);
  ~SiStripSpyMonitorModule();

 private:

  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();


  //check if contains pedsubtr data = 0
  bool hasNegativePedSubtr(const edm::DetSetVector<SiStripRawDigi>::detset & channelDigis,
			   uint16_t aPair);

  bool identifyTickmarks(const edm::DetSetVector<SiStripRawDigi>::detset & channelDigis,
			 const uint16_t threshold);

  edm::DetSetVector<SiStripRawDigi>::detset::const_iterator
  findTwoConsecutive(const edm::DetSetVector<SiStripRawDigi>::detset & channelDigis,
		     const uint16_t threshold,
		     uint16_t & aCounter);

  //tag of spydata collection
  edm::InputTag spyScopeRawDigisTag_;
  edm::InputTag spyPedSubtrDigisTag_;

  //tag of l1A and apveAddress counters
  edm::InputTag spyL1Tag_;
  edm::InputTag spyTotCountTag_;
  edm::InputTag spyAPVeTag_;

  uint32_t minDigiRange_;
  uint32_t maxDigiRange_;
  uint32_t minDigitalLow_;
  uint32_t maxDigitalLow_;
  uint32_t minDigitalHigh_;
  uint32_t maxDigitalHigh_;

  unsigned int evt_;

  DQMStore* dqm_;
  //folder name for histograms in DQMStore
  std::string folderName_;
  //book detailed histograms even if they will be empty (for merging)
  bool fillAllDetailedHistograms_;
  //do histos vs time with time=event number. Default time = orbit number (s)
  bool fillWithEvtNum_;
  bool fillWithLocalEvtNum_;
  //write the DQMStore to a root file at the end of the job
  bool writeDQMStore_;
  std::string dqmStoreFileName_;

  SPYHistograms histManager_;  
  uint16_t firstHeaderBit_;
  uint16_t firstTrailerBit_;

  sistrip::SpyUtilities utility_;
  sistrip::SpyUtilities::FrameQuality frameQuality_;

  std::ofstream outfile_[20];  
  std::vector<std::string> outfileNames_;  
  std::map<std::string,unsigned int> outfileMap_;

  bool writeCabling_;

};



using edm::LogError;
using edm::LogWarning;
using edm::LogInfo;
//
// Constructors and destructor
//

SiStripSpyMonitorModule::SiStripSpyMonitorModule(const edm::ParameterSet& iConfig)
  : spyScopeRawDigisTag_(iConfig.getUntrackedParameter<edm::InputTag>("SpyScopeRawDigisTag",edm::InputTag("SiStripSpyUnpacker","ScopeRawDigis"))),
    spyPedSubtrDigisTag_(iConfig.getUntrackedParameter<edm::InputTag>("SpyPedSubtrDigisTag",edm::InputTag("SiStripFEDEmulator",""))),
    spyL1Tag_(iConfig.getUntrackedParameter<edm::InputTag>("SpyL1Tag",edm::InputTag("SiStripSpyDigiConverter","L1ACount"))),
    spyTotCountTag_(iConfig.getUntrackedParameter<edm::InputTag>("SpyTotalEventCountTag",edm::InputTag("SiStripSpyDigiConverter","TotalEventCount"))),
    spyAPVeTag_(iConfig.getUntrackedParameter<edm::InputTag>("SpyAPVeTag",edm::InputTag("SiStripSpyDigiConverter","APVAddress"))),
    dqm_(0),
    folderName_(iConfig.getUntrackedParameter<std::string>("HistogramFolderName","SiStrip/ReadoutView/SpyMonitoringSummary")),
    fillAllDetailedHistograms_(iConfig.getUntrackedParameter<bool>("FillAllDetailedHistograms",false)),
    fillWithEvtNum_(iConfig.getUntrackedParameter<bool>("FillWithEventNumber",false)),
    fillWithLocalEvtNum_(iConfig.getUntrackedParameter<bool>("FillWithLocalEventNumber",false)),
    writeDQMStore_(iConfig.getUntrackedParameter<bool>("WriteDQMStore",false)),
    dqmStoreFileName_(iConfig.getUntrackedParameter<std::string>("DQMStoreFileName","DQMStore.root")),
    firstHeaderBit_(0),
    firstTrailerBit_(0),
    outfileNames_(iConfig.getUntrackedParameter<std::vector<std::string> >("OutputErrors")),
    writeCabling_(iConfig.getUntrackedParameter<bool>("WriteCabling",false))
{

  evt_ = 0;
  std::ostringstream pDebugStream;
  histManager_.initialise(iConfig,&pDebugStream);
  const unsigned int nFiles = outfileNames_.size();

  for (unsigned int i(0); i<nFiles; i++){
    std::ostringstream lName;
    lName << outfileNames_.at(i) <<  ".out";
    if (i<20) outfile_[i].open(lName.str().c_str(),std::ios::out);
    outfileMap_[outfileNames_.at(i)] = i;
  }

  frameQuality_.minDigiRange = static_cast<uint16_t>(iConfig.getUntrackedParameter<uint32_t>("MinDigiRange",100));
  frameQuality_.maxDigiRange = static_cast<uint16_t>(iConfig.getUntrackedParameter<uint32_t>("MaxDigiRange",1024));
  frameQuality_.minZeroLight = static_cast<uint16_t>(iConfig.getUntrackedParameter<uint32_t>("MinZeroLight",0));
  frameQuality_.maxZeroLight = static_cast<uint16_t>(iConfig.getUntrackedParameter<uint32_t>("MaxZeroLight",1024));
  frameQuality_.minTickHeight = static_cast<uint16_t>(iConfig.getUntrackedParameter<uint32_t>("MinTickHeight",0));
  frameQuality_.maxTickHeight = static_cast<uint16_t>(iConfig.getUntrackedParameter<uint32_t>("MaxTickHeight",1024));
  
}

SiStripSpyMonitorModule::~SiStripSpyMonitorModule()
{ 
  const unsigned int nFiles = outfileNames_.size();
  for (unsigned int i(0); i<nFiles; i++){
    outfile_[i].close();
  }

  outfileMap_.clear();
  outfileNames_.clear();

}


// ------------ method called once each job just before starting event loop  ------------
void 
SiStripSpyMonitorModule::beginJob()
{

  //get DQM store
  dqm_ = &(*edm::Service<DQMStore>());
  dqm_->setCurrentFolder(folderName_);

  LogInfo("SiStripSpyMonitorModule") << " Histograms will be written in " 
				     << folderName_ 
				     << ". Current folder is : " 
				     << dqm_->pwd() 
				     << std::endl;
  
  //this propagates dqm_ to the histoclass, must be called !
  histManager_.bookTopLevelHistograms(dqm_);
  
  if (fillAllDetailedHistograms_) histManager_.bookAllFEDHistograms();

  evt_ = 0;
  firstHeaderBit_ = 0;
  firstTrailerBit_ = 0;
}



// ------------ method called to for each event  ------------
void
SiStripSpyMonitorModule::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{

  static bool firstEvent = true;
  //update cabling and pedestals
  const SiStripFedCabling* lCabling = utility_.getCabling( iSetup );
  if (firstEvent && writeCabling_){
    std::ofstream lOutCabling;
    lOutCabling.open("trackerDetId_FEDIdChNum_list.txt",std::ios::out);
    for (uint16_t lFedId = sistrip::FED_ID_MIN; lFedId <= sistrip::FED_ID_MAX; ++lFedId) {//loop on feds
      for (uint16_t lFedChannel = 0; lFedChannel < sistrip::FEDCH_PER_FED; lFedChannel++){//loop on channels
	const FedChannelConnection & lConnection = lCabling->connection(lFedId,lFedChannel);
	if (!lConnection.isConnected()) continue;
	uint32_t lDetId = lConnection.detId();
	lOutCabling << "FED ID = " << lFedId 
		    << ", Channel = " << lFedChannel 
		    << ",fedkey = " << sistrip::FEDCH_PER_FED*lFedId + lFedChannel
		    << ", detId = " << lDetId 
		    << std::endl;
      }
    }
    lOutCabling.close();
  }

  //For spy data
  //get map of TotalEventCount and L1ID, indexed by fedId, and APVaddress indexed by fedIndex.
  edm::Handle<std::vector<uint32_t> > lSpyL1IDHandle,lSpyTotCountHandle,lSpyAPVeHandle;
  try {
    iEvent.getByLabel(spyL1Tag_,lSpyL1IDHandle);
    iEvent.getByLabel(spyTotCountTag_,lSpyTotCountHandle);
    iEvent.getByLabel(spyAPVeTag_,lSpyAPVeHandle);
  }
  catch (const cms::Exception& e) {
    LogError("SiStripSpyMonitorModule") << e.what() ;
    return;
  }
  //const std::map<uint32_t,uint32_t> & lSpyMaxCountMap = *lSpyL1IDHandle;
  //const std::map<uint32_t,uint32_t> & lSpyMinCountMap = *lSpyTotCountHandle;
  const std::vector<uint32_t> & lSpyAPVeVec = *lSpyAPVeHandle;

  //retrieve the scope digis
  edm::Handle<edm::DetSetVector<SiStripRawDigi> > digisHandle;
  try {
    iEvent.getByLabel(spyScopeRawDigisTag_, digisHandle);
  }
  catch (const cms::Exception& e) {
    LogError("SiStripSpyMonitorModule") << e.what() ;
    return;
  }
  const edm::DetSetVector<SiStripRawDigi> * lInputDigis = digisHandle.product();

  //retrieve the reordered payload digis
  edm::Handle<edm::DetSetVector<SiStripRawDigi> > payloadHandle;
  try {
    iEvent.getByLabel(spyPedSubtrDigisTag_, payloadHandle);
  }
  catch (const cms::Exception& e) {
    LogError("SiStripSpyMonitorModule") << e.what() ;
    return;
  }
  const edm::DetSetVector<SiStripRawDigi> * lPayloadDigis = payloadHandle.product();


  //for first event only
  //loop first on channels to calculate majority value of when the first header bit is found.
  //output info message to give the value found
  //should currently be 6 but may vary in the futur
  //then we can check firstTrailerBit is +256+24 after

  if (firstEvent){
    sistrip::SpyUtilities::getMajorityHeader(lInputDigis,firstHeaderBit_);
    firstTrailerBit_ = firstHeaderBit_+24+sistrip::STRIPS_PER_FEDCH;
  }

  //initialise some counters, filled in histos eventually
  SPYHistograms::ErrorCounters lCounters;
  lCounters.nNoData = 0;
  lCounters.nLowRange = 0;
  lCounters.nHighRange = 0;
  lCounters.nMinZero = 0;
  lCounters.nMaxSat = 0;
  lCounters.nLowPb = 0;
  lCounters.nHighPb = 0;
  lCounters.nOOS = 0;
  lCounters.nOtherPbs = 0;
  lCounters.nAPVError = 0;
  lCounters.nAPVAddressError = 0;
  lCounters.nNegPeds = 0;



  //fill event number for output text files
  const unsigned int nFiles = outfileNames_.size();
  for (unsigned int i(0); i<nFiles; i++){
    outfile_[i] << "**** evt "<< iEvent.id().event() << " ****" << std::endl ;
  }


  //loop over all FEDs and channels
  
  for (uint16_t lFedId = sistrip::FED_ID_MIN; lFedId <= sistrip::FED_ID_MAX; ++lFedId) {//loop on feds

    SPYHistograms::Errors lFEDErrors;
    lFEDErrors.hasNoData = false;
    lFEDErrors.hasLowRange = false; 
    lFEDErrors.hasHighRange = false; 
    lFEDErrors.hasMinZero = false;
    lFEDErrors.hasMaxSat = false;
    lFEDErrors.hasLowPb = false;
    lFEDErrors.hasHighPb = false;
    lFEDErrors.hasOOS = false;
    lFEDErrors.hasOtherPbs = false;
    lFEDErrors.hasErrorBit0 = false;
    lFEDErrors.hasErrorBit1 = false;
    lFEDErrors.hasAPVAddressError0 = false;
    lFEDErrors.hasAPVAddressError1 = false;
    lFEDErrors.hasNegPeds = false;
 
    uint32_t lAPVAddrRef = lSpyAPVeVec.at(lFedId);

    for (uint16_t lFedChannel = 0; lFedChannel < sistrip::FEDCH_PER_FED; lFedChannel++){//loop on channels
      
      uint32_t lFedIndex = sistrip::FEDCH_PER_FED*lFedId + lFedChannel;
      
      const FedChannelConnection & lConnection = lCabling->connection(lFedId,lFedChannel);

      if (!lConnection.isConnected()) continue;

      uint32_t lDetId = lConnection.detId();
      //uint16_t lNPairs = lConnection.nApvPairs();
      uint16_t lPair = lConnection.apvPairNumber();

      edm::DetSetVector<SiStripRawDigi>::const_iterator lDigis = lInputDigis->find(lFedIndex);

      //pedsubtr digis
      edm::DetSetVector<SiStripRawDigi>::const_iterator lPayload = lPayloadDigis->find(lDetId);
       
      //no digis found, continue.
      if (lDigis == lInputDigis->end()) {
	LogDebug("SiStripSpyMonitorModule") << " -- digis not found in ScopeRawDigis map for FEDID " 
						  << lFedId << " and FED channel " << lFedChannel << std::endl;
	continue;
      }

      sistrip::SpyUtilities::Frame lFrame = sistrip::SpyUtilities::extractFrameInfo(*lDigis);
      
      SPYHistograms::Errors lErrors;
      lErrors.hasNoData = false;
      lErrors.hasLowRange = false; 
      lErrors.hasHighRange = false; 
      lErrors.hasMinZero = false;
      lErrors.hasMaxSat = false;
      lErrors.hasLowPb = false;
      lErrors.hasHighPb = false;
      lErrors.hasOOS = false;
      lErrors.hasOtherPbs = false;
      lErrors.hasErrorBit0 = false;
      lErrors.hasErrorBit1 = false;
      lErrors.hasAPVAddressError0 = false;
      lErrors.hasAPVAddressError1 = false;
      lErrors.hasNegPeds = false;
      
      uint16_t lRange = sistrip::SpyUtilities::range(lFrame);
      uint16_t lThreshold = sistrip::SpyUtilities::threshold(lFrame);

      if (lRange == 0) {
	lCounters.nNoData++;
	lErrors.hasNoData = true;
 	lFEDErrors.hasNoData = true;
	if (outfileMap_.find("NoData") != outfileMap_.end()) outfile_[outfileMap_["NoData"]] << lFedId << " " << lFedChannel << " " << lDetId << std::endl;
      }
      else if (lFrame.digitalLow == 0 && lRange > 0) {
	lCounters.nMinZero++;
	lErrors.hasMinZero = true;
 	lFEDErrors.hasMinZero = true;
	if (outfileMap_.find("MinZero") != outfileMap_.end()) outfile_[outfileMap_["MinZero"]] << lFedId << " " << lFedChannel << " " << lDetId << std::endl;
      }
      else if (lFrame.digitalHigh >= 1023){
	lCounters.nMaxSat++;
	lErrors.hasMaxSat = true;
 	lFEDErrors.hasMaxSat = true;
	if (outfileMap_.find("MaxSat") != outfileMap_.end()) outfile_[outfileMap_["MaxSat"]]  << lFedId << " " << lFedChannel << " " << lDetId << std::endl;
      }
      else if (lRange > 0 && lRange < frameQuality_.minDigiRange) {
	lCounters.nLowRange++;
	lErrors.hasLowRange = true;
 	lFEDErrors.hasLowRange = true;
	if (outfileMap_.find("LowRange") != outfileMap_.end()) outfile_[outfileMap_["LowRange"]]  << lFedId << " " << lFedChannel << " " << lDetId << std::endl;
      }
      else if (lRange > frameQuality_.maxDigiRange) {
	lCounters.nHighRange++;
	lErrors.hasHighRange = true;
 	lFEDErrors.hasHighRange = true;
	if (outfileMap_.find("HighRange") != outfileMap_.end()) outfile_[outfileMap_["HighRange"]]  << lFedId << " " << lFedChannel << " " << lDetId << std::endl;
      }
      else if (lFrame.digitalLow < frameQuality_.minZeroLight || lFrame.digitalLow > frameQuality_.maxZeroLight) {
	lCounters.nLowPb++;
	lErrors.hasLowPb = true;
 	lFEDErrors.hasLowPb = true;
	if (outfileMap_.find("LowPb") != outfileMap_.end()) outfile_[outfileMap_["LowPb"]]  << lFedId << " " << lFedChannel << " " << lDetId << std::endl;
      }
      else if (lFrame.digitalHigh < frameQuality_.minTickHeight || lFrame.digitalHigh > frameQuality_.maxTickHeight){
	lCounters.nHighPb++;
	lErrors.hasHighPb = true;
 	lFEDErrors.hasHighPb = true;
	if (outfileMap_.find("HighPb") != outfileMap_.end()) outfile_[outfileMap_["HighPb"]]  << lFedId << " " << lFedChannel << " " << lDetId << std::endl;
      }
      else if ( lFrame.firstHeaderBit != firstHeaderBit_ && //header in wrong position
		( (lFrame.firstHeaderBit != sistrip::SPY_SAMPLES_PER_CHANNEL && //header and 
		   lFrame.firstTrailerBit != sistrip::SPY_SAMPLES_PER_CHANNEL && //trailer found
		   lFrame.firstTrailerBit-lFrame.firstHeaderBit == 280) ||      //+ right distance between them
		  (lFrame.firstHeaderBit != sistrip::SPY_SAMPLES_PER_CHANNEL && // or header found
		   lFrame.firstTrailerBit == sistrip::SPY_SAMPLES_PER_CHANNEL && // and trailer not found
		   lFrame.firstHeaderBit > 16 ) || // corresponding to back-to-back frame late enough
		  (lFrame.firstHeaderBit == sistrip::SPY_SAMPLES_PER_CHANNEL && // or header not found
		   identifyTickmarks(*lDigis,lThreshold) ) // but such that tickmark compatible with OOS frame
		  )
		){
	lCounters.nOOS++;
	lErrors.hasOOS = true;
 	lFEDErrors.hasOOS = true;
	if (outfileMap_.find("OOS") != outfileMap_.end()) outfile_[outfileMap_["OOS"]]  << lFedId << " " << lFedChannel << " " << lDetId << std::endl;
      }
      else if ( !(lFrame.firstHeaderBit == firstHeaderBit_ && lFrame.firstTrailerBit == firstTrailerBit_) ){
	lCounters.nOtherPbs++;
	lErrors.hasOtherPbs = true;
 	lFEDErrors.hasOtherPbs = true;
	if (outfileMap_.find("OtherPbs") != outfileMap_.end()) outfile_[outfileMap_["OtherPbs"]]  << lFedId << " " << lFedChannel << " " << lDetId << std::endl;
      }
      else if (lFrame.apvErrorBit.first || lFrame.apvErrorBit.second) {
	if (lFrame.apvErrorBit.first) {
	  lCounters.nAPVError++;
	  lErrors.hasErrorBit0 = true;
	  lFEDErrors.hasErrorBit0 = true;
	}
	if (lFrame.apvErrorBit.second) {
	  lCounters.nAPVError++;
	  lErrors.hasErrorBit1 = true;
	  lFEDErrors.hasErrorBit1 = true;
	}
	if (outfileMap_.find("APVError") != outfileMap_.end()) {
	  outfile_[outfileMap_["APVError"]]  << lFedId << " " << lFedChannel << " " << lDetId ;
	  if (lFrame.apvErrorBit.first) outfile_[outfileMap_["APVError"]] << " APV0" << std::endl;
	  if (lFrame.apvErrorBit.second) outfile_[outfileMap_["APVError"]] << " APV1" << std::endl;
	}
      }
      else if ( lFrame.apvAddress.first != lAPVAddrRef || 
		lFrame.apvAddress.second != lAPVAddrRef ) {
	if (lFrame.apvAddress.first != lAPVAddrRef){
	  lCounters.nAPVAddressError++;
	  lErrors.hasAPVAddressError0 = true;
	  lFEDErrors.hasAPVAddressError0 = true;
	}
	if (lFrame.apvAddress.second != lAPVAddrRef){
	  lCounters.nAPVAddressError++;
	  lErrors.hasAPVAddressError1 = true;
	  lFEDErrors.hasAPVAddressError1 = true;
	}
	if (outfileMap_.find("APVAddressError") != outfileMap_.end()){
	  outfile_[outfileMap_["APVAddressError"]]  << lFedId << " " << lFedChannel << " " << lDetId << std::endl;
	  if (lFrame.apvAddress.first != lAPVAddrRef) outfile_[outfileMap_["APVAddressError"]]  << " APV0" << std::endl;
	  if (lFrame.apvAddress.second != lAPVAddrRef) outfile_[outfileMap_["APVAddressError"]]  << " APV1" << std::endl;
	}
      }
      else if (lPayload != lPayloadDigis->end() && hasNegativePedSubtr(*lPayload,lPair)){
	lCounters.nNegPeds++;
	lErrors.hasNegPeds = true;
 	lFEDErrors.hasNegPeds = true;
	if (outfileMap_.find("NegPeds") != outfileMap_.end()) outfile_[outfileMap_["NegPeds"]]  << lFedId << " " << lFedChannel << " " << lDetId << std::endl;
      }

      histManager_.fillDetailedHistograms(lErrors,lFrame,lFedId,lFedChannel);

    }//loop on channels


    histManager_.fillFEDHistograms(lFEDErrors,lFedId);

  }//loop on feds

  double lTime;
  //if (fillWithEvtNum_) 
  //lTime = iEvent.id().event();
  //else if (fillWithLocalEvtNum_) lTime = evt_;
  //no orbit number for spy data !!
  //else lTime = iEvent.orbitNumber()/11223.;
  lTime = iEvent.id().event();
  if (fillWithLocalEvtNum_) lTime = evt_;
  
  histManager_.fillCountersHistograms(lCounters,lTime);
 

  //used to fill histo vs time with local event number....
  evt_++;
  firstEvent = false;

}//analyze method



// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripSpyMonitorModule::endJob()
{

  LogInfo("SiStripSpyMonitorModule") << "WriteDQMStore ? " << writeDQMStore_ 
				     << " file name = " << dqmStoreFileName_
				     << std::endl;
  if (writeDQMStore_) dqm_->save(dqmStoreFileName_);

  const unsigned int nFiles = outfileNames_.size();
  for (unsigned int i(0); i<nFiles; i++){
    outfile_[i].close();
  }

  outfileMap_.clear();
  outfileNames_.clear();

}



bool SiStripSpyMonitorModule::hasNegativePedSubtr(const edm::DetSetVector<SiStripRawDigi>::detset & channelDigis,
						  uint16_t aPair)
{
  edm::DetSetVector<SiStripRawDigi>::detset::const_iterator iDigi = channelDigis.begin();
  const edm::DetSetVector<SiStripRawDigi>::detset::const_iterator endChannelDigis = channelDigis.end();

  uint32_t count = 0;
  for (; iDigi != endChannelDigis; ++iDigi) {
    const uint16_t val = iDigi->adc();
    uint16_t lPair = static_cast<uint16_t>(count/sistrip::STRIPS_PER_FEDCH);
    if (val == 0 && lPair == aPair) return true;
    count++;
  }

  return false;

}

bool SiStripSpyMonitorModule::identifyTickmarks(const edm::DetSetVector<SiStripRawDigi>::detset & channelDigis,
						      const uint16_t threshold) 
{

  //start from the end
  uint16_t count = sistrip::SPY_SAMPLES_PER_CHANNEL-3;
  uint16_t lastPos = sistrip::SPY_SAMPLES_PER_CHANNEL;
  uint16_t nTrailers = 0;
  edm::DetSetVector<SiStripRawDigi>::detset::const_iterator iDigi = channelDigis.end();

  for ( ; count == 0; count--){
    iDigi = findTwoConsecutive(channelDigis,threshold,count);
    //if found, in different position = 70 before than previous value, go and look 70 before
    if (iDigi != channelDigis.end() && 
	(lastPos == sistrip::SPY_SAMPLES_PER_CHANNEL || count == lastPos+1-70)) {
      nTrailers++;
      lastPos = count-1;
      count -= 70;
    }
    //else keep looking
    else count--;
  }

  if (nTrailers > 1) LogDebug("SiStripSpyMonitorModule") << " -- Found "
							 << nTrailers << " trailers every 70 clock cycles for channel " 
							 << channelDigis.detId() 
							 << ", evt " << evt_
							 << std::endl;
  //if only one found, should be < 280 otherwise header should have been found and this method would not be called
  return (nTrailers > 1) || (nTrailers == 1 && lastPos < 280);

}

edm::DetSetVector<SiStripRawDigi>::detset::const_iterator
SiStripSpyMonitorModule::findTwoConsecutive(const edm::DetSetVector<SiStripRawDigi>::detset & channelDigis,
						  const uint16_t threshold,
						  uint16_t & aCounter)
{
  const edm::DetSetVector<SiStripRawDigi>::detset::const_iterator endChannelDigis = channelDigis.end();
  edm::DetSetVector<SiStripRawDigi>::detset::const_iterator lStart = channelDigis.begin()+aCounter;

  bool foundTrailer = false;
  // Loop over digis looking for last two above threshold
  uint8_t aboveThreshold = 0;

  for (; lStart != endChannelDigis; ++lStart) {
    if ( lStart->adc() > threshold) {
      aboveThreshold++;
    }
    else {
      aboveThreshold = 0;
    }
    if (aboveThreshold == 2) {foundTrailer = true; break; }
    aCounter++;
  }//end of loop over digis
  
  if (foundTrailer) return lStart;
  else {
    aCounter = sistrip::SPY_SAMPLES_PER_CHANNEL;
    return endChannelDigis;
  }
}

//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripSpyMonitorModule);

