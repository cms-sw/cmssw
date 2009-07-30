
// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      SiStripCMMonitorPlugin
// 
/**\class SiStripCMMonitorPlugin SiStripCMMonitor.cc DQM/SiStripMonitorHardware/plugins/SiStripCMMonitor.cc

 Description: DQM source application to monitor common mode for SiStrip data
*/
//
//         Created:  2009/07/22
// $Id: SiStripCMMonitor.cc,v 1.3 2009/07/24 06:56:29 amagnan Exp $
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

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "DPGAnalysis/SiStripTools/interface/APVShotFinder.h"
#include "DPGAnalysis/SiStripTools/interface/APVShot.h"

#include "DQM/SiStripMonitorHardware/interface/CMHistograms.hh"

//
// Class declaration
//

class SiStripCMMonitorPlugin : public edm::EDAnalyzer
{
 public:

  explicit SiStripCMMonitorPlugin(const edm::ParameterSet&);
  ~SiStripCMMonitorPlugin();
 private:
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  //update the cabling if necessary
  void updateCabling(const edm::EventSetup& eventSetup);

  //medians of all strip per channel (APV1,APV2). Return 0,0 if not zero-suppressed mode.
  std::pair<uint16_t,uint16_t> fillMedians(const std::string & aMode,
					   const sistrip::FEDChannel & aChannel,
					   const unsigned int aIndex
					   );
  //tag of FEDRawData collection
  edm::InputTag rawDataTag_;
  //folder name for histograms in DQMStore
  std::string folderName_;
  //book detailed histograms even if they will be empty (for merging)
  bool fillAllDetailedHistograms_;
  //do histos vs time with time=event number. Default time = orbit number (s)
  bool fillWithEvtNum_;
  //print debug messages when problems are found: 1=error debug, 2=light debug, 3=full debug
  unsigned int printDebug_;
  //write the DQMStore to a root file at the end of the job
  bool writeDQMStore_;
  std::string dqmStoreFileName_;
  //the DQMStore
  DQMStore* dqm_;
  //FED cabling
  uint32_t cablingCacheId_;
  const SiStripFedCabling* cabling_;

  //add parameter to save computing time if TkHistoMap are not filled
  bool doTkHistoMap_;

  edm::InputTag _digicollection;
  bool _zs;

  CMHistograms cmHists_;

  std::map<unsigned int,std::pair<double,double> > CommonModesAPV0_;
  std::map<unsigned int,std::pair<double,double> > CommonModesAPV1_;
  unsigned int NEntries_[2];


};


//
// Constructors and destructor
//

SiStripCMMonitorPlugin::SiStripCMMonitorPlugin(const edm::ParameterSet& iConfig)
  : rawDataTag_(iConfig.getUntrackedParameter<edm::InputTag>("RawDataTag",edm::InputTag("source",""))),
    folderName_(iConfig.getUntrackedParameter<std::string>("HistogramFolderName","SiStrip/ReadoutView/CMMonitoring")),
    fillAllDetailedHistograms_(iConfig.getUntrackedParameter<bool>("FillAllDetailedHistograms",false)),
    fillWithEvtNum_(iConfig.getUntrackedParameter<bool>("FillWithEventNumber",false)),
    printDebug_(iConfig.getUntrackedParameter<unsigned int>("PrintDebugMessages",1)),
    writeDQMStore_(iConfig.getUntrackedParameter<bool>("WriteDQMStore",false)),
    dqmStoreFileName_(iConfig.getUntrackedParameter<std::string>("DQMStoreFileName","DQMStore.root")),
    cablingCacheId_(0),
    _digicollection(iConfig.getParameter<edm::InputTag>("digiCollection")),
    _zs(iConfig.getUntrackedParameter<bool>("zeroSuppressed",true))
 
{
  //print config to debug log
  std::ostringstream debugStream;
  if (printDebug_>1) {
    debugStream << "[SiStripCMMonitorPlugin]Configuration for SiStripCMMonitorPlugin: " << std::endl
                << "[SiStripCMMonitorPlugin]\tRawDataTag: " << rawDataTag_ << std::endl
                << "[SiStripCMMonitorPlugin]\tHistogramFolderName: " << folderName_ << std::endl
                << "[SiStripCMMonitorPlugin]\tFillAllDetailedHistograms? " << (fillAllDetailedHistograms_ ? "yes" : "no") << std::endl
		<< "[SiStripCMMonitorPlugin]\tFillWithEventNumber?" << (fillWithEvtNum_ ? "yes" : "no") << std::endl
                << "[SiStripCMMonitorPlugin]\tPrintDebugMessages? " << (printDebug_ ? "yes" : "no") << std::endl
                << "[SiStripCMMonitorPlugin]\tWriteDQMStore? " << (writeDQMStore_ ? "yes" : "no") << std::endl;
    if (writeDQMStore_) debugStream << "[SiStripCMMonitorPlugin]\tDQMStoreFileName: " << dqmStoreFileName_ << std::endl;
  }
    
 std::ostringstream* pDebugStream = (printDebug_>1 ? &debugStream : NULL);

 cmHists_.initialise(iConfig,pDebugStream);

 doTkHistoMap_ = cmHists_.isTkHistoMapEnabled(cmHists_.tkHistoMapName());


 if (printDebug_) {
   LogTrace("SiStripMonitorHardware") << debugStream.str();
 }


}

SiStripCMMonitorPlugin::~SiStripCMMonitorPlugin()
{
}


//
// Member functions
//

// ------------ method called to for each event  ------------
void
SiStripCMMonitorPlugin::analyze(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{

  static bool firstEvent = true;
  //update cabling
  updateCabling(iSetup);
  
  //get raw data
  edm::Handle<FEDRawDataCollection> rawDataCollectionHandle;
  iEvent.getByLabel(rawDataTag_,rawDataCollectionHandle);
  const FEDRawDataCollection& rawDataCollection = *rawDataCollectionHandle;
  
  //get digi data
  edm::Handle<edm::DetSetVector<SiStripDigi> > digis;
  iEvent.getByLabel(_digicollection,digis);

  // loop on detector with digis

  APVShotFinder apvsf(*digis,_zs);
  const std::vector<APVShot>& shots = apvsf.getShots();

  //FED errors
  FEDErrors lFedErrors;

  unsigned int lNMonitoring = 0;
  unsigned int lNUnpacker = 0;
  unsigned int lNTotBadFeds = 0;

  //initialise map of fedId/mean+RMS CM
  std::pair<std::map<unsigned int,std::pair<double,double> >::iterator,bool> alreadyThere[2];

  //loop over siStrip FED IDs
  for (unsigned int fedId = FEDNumbering::MINSiStripFEDID; 
       fedId <= FEDNumbering::MAXSiStripFEDID; 
       fedId++) {//loop over FED IDs
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);

    //create an object to fill all errors
    lFedErrors.initialise(fedId,cabling_);

    bool lFullDebug = false;

    //Do detailed check
    //first check if data exists
    bool lDataExist = lFedErrors.checkDataPresent(fedData);
    if (!lDataExist) {
      continue;
    }

    //Do exactly same check as unpacker
    //will be used by channel check in following method fillFEDErrors so need to be called beforehand.
    bool lFailUnpackerFEDcheck = lFedErrors.failUnpackerFEDCheck(fedData);
 
    //check for problems
    lFedErrors.fillFEDErrors(fedData,
			     lFullDebug,
			     printDebug_
			     );

    bool lFailMonitoringFEDcheck = lFedErrors.failMonitoringFEDCheck();
    if (lFailMonitoringFEDcheck) lNTotBadFeds++;

   
    //sanity check: if something changed in the unpacking code 
    //but wasn't propagated here
    if (lFailMonitoringFEDcheck != lFailUnpackerFEDcheck && printDebug_) {
      std::ostringstream debugStream;
      debugStream << " --- WARNING: FED " << fedId << std::endl 
		  << " ------ Monitoring FED check " ;
      if (lFailMonitoringFEDcheck) debugStream << "failed." << std::endl;
      else debugStream << "passed." << std::endl ;
      debugStream << " ------ Unpacker FED check " ;
      if (lFailUnpackerFEDcheck) debugStream << "failed." << std::endl;
      else debugStream << "passed." << std::endl ;

      if (lFailMonitoringFEDcheck) lNMonitoring++;
      else if (lFailUnpackerFEDcheck) lNUnpacker++;
      edm::LogError("SiStripMonitorHardware") << debugStream.str();

    }

    if (lFedErrors.failMonitoringFEDCheck() ||
	lFedErrors.anyFEProblems()) continue;

    std::ostringstream infoStream;

    
    if (printDebug_ > 1) {
      infoStream << " --- Processing FED #" << fedId << std::endl;
    }

    //need to construct full object to go any further
    std::auto_ptr<const sistrip::FEDBuffer> buffer;
    buffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size(),true));


    std::vector<CMHistograms::CMvalues> values;

    for (unsigned int iCh = 0; 
	 iCh < sistrip::FEDCH_PER_FED; 
	 iCh++) {//loop on channels

      const FedChannelConnection & lConnection = cabling_->connection(fedId,iCh);
      bool connected = lConnection.isConnected();

      //std::cout << "FedID " << fedId << ", ch " << iCh << ", nAPVPairs " << lConnection.nApvPairs() << " apvPairNumber " << lConnection.apvPairNumber() << std::endl;

      if (!connected) continue;

      bool isBadChan = false;
      for (unsigned int badCh(0); badCh<lFedErrors.getBadChannels().size(); badCh++) {
	if (lFedErrors.getBadChannels().at(badCh).first == iCh) {
	  isBadChan = true;
	  break;
	}
      }

      if (isBadChan) continue;

      uint32_t lDetId = lConnection.detId();
      short lAPVPair = lConnection.apvPairNumber();
      short lSubDet = 0;
      std::pair<float,float> lShotMedian = std::pair<float,float>(0,0);

      bool isShot = false;
      //bool isFirst = true;
      for(std::vector<APVShot>::const_iterator shot=shots.begin();shot!=shots.end();++shot) {

	if (shot->detId() == lDetId && static_cast<short>(shot->apvNumber()/2.) == lAPVPair) {
	  if(shot->isGenuine()) {
	    lSubDet = shot->subDet();
	    isShot = true;
	    if (shot->apvNumber()%2 == 0) lShotMedian.first = shot->median();
	    else if (shot->apvNumber()%2 == 1) lShotMedian.second = shot->median();
	    //shot->nStrips()
	  }
	  //isFirst = false;
	}
	else {
	  if (isShot) break;
	}
      }

      //if (!isShot) continue;

      if (isShot && printDebug_ > 2){
	const sistrip::FEDBufferBase* debugBuffer = NULL;
	std::auto_ptr<const sistrip::FEDBufferBase> bufferBase;
	bufferBase.reset(new sistrip::FEDBufferBase(fedData.data(),fedData.size()));
	std::ostringstream debugStream;
	debugStream << "Found shot for FedID " << fedId << ", channel " << iCh << std::endl;
	if (buffer.get()) debugBuffer = buffer.get();
	else if (bufferBase.get()) debugBuffer = bufferBase.get();
	if (debugBuffer) {
	  debugStream << (*debugBuffer) << std::endl;
	  debugBuffer->dump(debugStream);
	  debugStream << std::endl;
	  edm::LogInfo("SiStripMonitorHardware") << debugStream.str();
	}
      }


      if (firstEvent){
	if (lSubDet == 3) {
	  TIBDetId lId(lDetId);
	  std::cout << "TIB layer " << lId.layer()  << ", fedID " << fedId << ", channel " << iCh << std::endl;
	}
	else if (lSubDet == 4) {
	  TIDDetId lId(lDetId);
	  std::cout << "TID side " << lId.side()  << " wheel " << lId.wheel() << ", ring " << lId.ring() << ", fedID " << fedId << ", channel " << iCh << std::endl;
	}
	else if (lSubDet == 5) {
	  TOBDetId lId(lDetId);
	  std::cout << "TOB side " << lId.rod()[0]  << " layer " << lId.layer() << ", rod " << lId.rodNumber() << ", fedID " << fedId << ", channel " << iCh << std::endl;
	}
	else if (lSubDet == 6) {
	  TECDetId lId(lDetId);
	  std::cout << "TEC side " << lId.side()  << " wheel " << lId.wheel() << ", petal " << lId.petalNumber() << ", ring " << lId.ring() << ", fedID " << fedId << ", channel " << iCh << std::endl;
	}
      }

      std::ostringstream lMode;
      lMode << buffer->readoutMode();
      const sistrip::FEDChannel & lChannel = buffer->channel(iCh);
      std::pair<uint16_t,uint16_t> medians = fillMedians(lMode.str(),lChannel,iCh);
      
      CMHistograms::CMvalues lVal;
      lVal.IsShot = isShot;
      lVal.ChannelID = iCh;
      lVal.Length = lChannel.length();
      lVal.Medians = std::pair<uint16_t,uint16_t>(medians.first,medians.second);
      lVal.ShotMedians = std::pair<float,float>(lShotMedian.first,lShotMedian.second);


      if (printDebug_ > 1) {
	if (lChannel.length() > 7) {
	  infoStream << "Medians for channel #" << iCh << " (length " << lChannel.length() << "): " << medians.first << ", " << medians.second << std::endl;
	}
      }

      //if some clusters are found:
      values.push_back(lVal);

      if (doTkHistoMap_){//if TkHistMap is enabled

	alreadyThere[0] = CommonModesAPV0_.insert(std::pair<unsigned int,std::pair<double,double> >(lDetId,std::pair<double,double>(medians.first,medians.first*medians.first)));
	if (alreadyThere[0].second) NEntries_[0]++;
	else {
	  ((alreadyThere[0].first)->second).first += medians.first;
	  ((alreadyThere[0].first)->second).second += medians.first*medians.first;
	  //nBadChans++;
	  NEntries_[0]++;
	}
	alreadyThere[1] = CommonModesAPV1_.insert(std::pair<unsigned int,std::pair<double,double> >(lDetId,std::pair<double,double>(medians.second,medians.second*medians.second)));
	if (alreadyThere[1].second) NEntries_[1]++;
	else {
	  ((alreadyThere[1].first)->second).first += medians.second;
	  ((alreadyThere[1].first)->second).second += medians.second*medians.second;
	  //nBadChans++;
	  NEntries_[1]++;
	}

      }

    }//loop on channels
    
    float lTime = 0;
    if (fillWithEvtNum_) lTime = iEvent.id().event();
    else lTime = iEvent.orbitNumber()/11223.;

    cmHists_.fillHistograms(values,lTime,fedId);

    if (printDebug_ > 1) edm::LogInfo("SiStripMonitorHardware") << infoStream.str();


  }//loop on FEDs



  if ((lNMonitoring > 0 || lNUnpacker > 0) && printDebug_) {
    std::ostringstream debugStream;
    debugStream
      << "[SiStripFEDMonitorPlugin]-------------------------------------------------------------------------" << std::endl 
      << "[SiStripFEDMonitorPlugin]-------------------------------------------------------------------------" << std::endl 
      << "[SiStripFEDMonitorPlugin]-- Summary of differences between unpacker and monitoring at FED level : " << std::endl 
      << "[SiStripFEDMonitorPlugin] ---- Number of times monitoring fails but not unpacking = " << lNMonitoring << std::endl 
      << "[SiStripFEDMonitorPlugin] ---- Number of times unpacking fails but not monitoring = " << lNUnpacker << std::endl
      << "[SiStripFEDMonitorPlugin]-------------------------------------------------------------------------" << std::endl 
      << "[SiStripFEDMonitorPlugin]-------------------------------------------------------------------------" << std::endl ;
    edm::LogError("SiStripMonitorHardware") << debugStream.str();

  }

  firstEvent = false;

}//analyze method

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripCMMonitorPlugin::beginJob(const edm::EventSetup&)
{
  //get DQM store
  dqm_ = &(*edm::Service<DQMStore>());
  dqm_->setCurrentFolder(folderName_);

  cmHists_.bookTopLevelHistograms(dqm_);

  if (fillAllDetailedHistograms_) cmHists_.bookAllFEDHistograms();

}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripCMMonitorPlugin::endJob()
{

  if (doTkHistoMap_) {//if TkHistoMap is enabled
    std::map<unsigned int,std::pair<double,double> >::iterator fracIter;
    std::vector<std::pair<unsigned int,unsigned int> >::iterator chanIter;

    //int ele = 0;
    //int nBadChannels = 0;
    for (fracIter = CommonModesAPV0_.begin(); fracIter!=CommonModesAPV0_.end(); fracIter++){
      uint32_t detid = fracIter->first;
      //if ((fracIter->second).second != 0) {
      //std::cout << "------ ele #" << ele << ", Frac for detid #" << detid << " = " <<(fracIter->second).second << "/" << (fracIter->second).first << std::endl;
      //nBadChannels++;
      //}
      float mean = (fracIter->second).first/NEntries_[0];
      float rms = sqrt((fracIter->second).second/(NEntries_[0]-1)-(mean*mean));
      cmHists_.fillTkHistoMap(cmHists_.tkHistoMapPointer(0),detid,mean);
      cmHists_.fillTkHistoMap(cmHists_.tkHistoMapPointer(1),detid,rms);
      //ele++;
    }

    for (fracIter = CommonModesAPV1_.begin(); fracIter!=CommonModesAPV1_.end(); fracIter++){
      uint32_t detid = fracIter->first;
      //if ((fracIter->second).second != 0) {
      //std::cout << "------ ele #" << ele << ", Frac for detid #" << detid << " = " <<(fracIter->second).second << "/" << (fracIter->second).first << std::endl;
      //nBadChannels++;
      //}
      float mean = (fracIter->second).first/NEntries_[1];
      float rms = sqrt((fracIter->second).second/(NEntries_[1]-1)-(mean*mean));
      cmHists_.fillTkHistoMap(cmHists_.tkHistoMapPointer(2),detid,mean);
      cmHists_.fillTkHistoMap(cmHists_.tkHistoMapPointer(3),detid,rms);
      //ele++;
    }

  }//if TkHistoMap is enabled

  if (writeDQMStore_) dqm_->save(dqmStoreFileName_);
}

void SiStripCMMonitorPlugin::updateCabling(const edm::EventSetup& eventSetup)
{
  uint32_t currentCacheId = eventSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (cablingCacheId_ != currentCacheId) {
    edm::ESHandle<SiStripFedCabling> cablingHandle;
    eventSetup.get<SiStripFedCablingRcd>().get(cablingHandle);
    cabling_ = cablingHandle.product();
    cablingCacheId_ = currentCacheId;
  }
}


std::pair<uint16_t,uint16_t> SiStripCMMonitorPlugin::fillMedians(const std::string & aMode,
								  const sistrip::FEDChannel & aChannel,
								  const unsigned int aIndex
								  )
{

  /// create unpacker only if zero-suppressed mode
  std::ostringstream debugStream;
  if (printDebug_ > 1) debugStream << "Readout mode = " << aMode << std::endl;
  if (aMode.find("Zero suppressed") == aMode.npos || aMode.find("lite") != aMode.npos) return std::pair<uint16_t,uint16_t>(0,0);

  const uint8_t* lData = aChannel.data();
  //data are organised by lines of 8 8-bit words, numbered from 0->7 then 8->15, etc...
  //word7 = beginning of packet for fiber12 = channel0, and is fibre12_len[0:7]
  //word6 =  fibre12_len[11:8]
  //if channel0 has no clusters (channel length=7), then offset for channel 1 is 7,
  //and word0=beginning of packet for fiber11 = channel1, and is fibre11_len[0:7].
  //the words should be inverted per line (offset^7: exclusive bit OR with 0x00000111):
  //7=0, 0=7, 1=6, 6=1, 8=15, 15=8, etc....
  uint8_t lWord1 = lData[aChannel.offset()^7];
  uint8_t lWord2 = lData[(aChannel.offset()+1)^7] & 0x0F;
  //uint8_t lWord3 = lData[(aChannel.offset()+2)^7];
  uint8_t lWord4 = lData[(aChannel.offset()+3)^7];
  uint8_t lWord5 = lData[(aChannel.offset()+4)^7] & 0x03;
  uint8_t lWord6 = lData[(aChannel.offset()+5)^7];
  uint8_t lWord7 = lData[(aChannel.offset()+6)^7] & 0x03;

  uint16_t lLength  = lWord1 + (lWord2 << 8);
  uint16_t lMedian0 = lWord4 + (lWord5 << 8);
  uint16_t lMedian1 = lWord6 + (lWord7 << 8);

  if (lLength != aChannel.length()) {
    if (printDebug_ > 1) debugStream << "Channel #" << aIndex << " offset: " << aChannel.offset() << ", offset^7 = " << (aChannel.offset()^7) << std::endl;
    if (printDebug_ > 1) debugStream << "My length = " << lLength << ", Nicks's length = " << aChannel.length() << std::endl;
  }

  if (printDebug_ > 1) edm::LogError("SiStripMonitorHardware") << debugStream.str();

  return std::pair<uint16_t,uint16_t>(lMedian0,lMedian1);
}

//
// Define as a plug-in
//






#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripCMMonitorPlugin);
