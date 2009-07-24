// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      SiStripFEDMediansPlugin
// 
/**\class SiStripFEDMediansPlugin SiStripFEDMedians.cc DQM/SiStripMonitorHardware/plugins/SiStripFEDMedians.cc

 Description: DQM source application to monitor common mode for SiStrip data
*/
//
//         Created:  2009/07/22
// $Id: SiStripFEDMedians.cc,v 1.2 2009/07/23 10:40:07 amagnan Exp $
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

#include "DQM/SiStripMonitorHardware/interface/FEDHistograms.hh"

//
// Class declaration
//

class SiStripFEDMediansPlugin : public edm::EDAnalyzer
{
 public:

  explicit SiStripFEDMediansPlugin(const edm::ParameterSet&);
  ~SiStripFEDMediansPlugin();
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

  void bookFEDHistograms(unsigned int fedId);

  //tag of FEDRawData collection
  edm::InputTag rawDataTag_;
  //folder name for histograms in DQMStore
  std::string folderName_;
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


  MonitorElement *medianAPV0_;
  MonitorElement *medianAPV1_;
  MonitorElement *shotMedianAPV0_;
  MonitorElement *shotMedianAPV1_;
  MonitorElement *medianAPV1vsAPV0_;
  MonitorElement *medianAPV1minusAPV0_;
  MonitorElement *diffMedianminusShotMedianAPV1_;
  MonitorElement *medianAPV0minusShot0_;
  MonitorElement *medianAPV1minusShot1_;
  MonitorElement *medianAPV1minusAPV0perFED_[FEDNumbering::MAXSiStripFEDID+1];
  MonitorElement *medianAPV1minusAPV0vsChperFED_[FEDNumbering::MAXSiStripFEDID+1];
  MonitorElement *medianAPV1minusAPV0vsTimeperFED_[FEDNumbering::MAXSiStripFEDID+1];
  MonitorElement *medianAPV0vsTimeperFED_[FEDNumbering::MAXSiStripFEDID+1];
  MonitorElement *medianAPV1vsTimeperFED_[FEDNumbering::MAXSiStripFEDID+1];

  edm::InputTag _digicollection;
  bool _zs;

  FEDHistograms::HistogramConfig timeConfig_;


};


//
// Constructors and destructor
//

SiStripFEDMediansPlugin::SiStripFEDMediansPlugin(const edm::ParameterSet& iConfig)
  : rawDataTag_(iConfig.getUntrackedParameter<edm::InputTag>("RawDataTag",edm::InputTag("source",""))),
    folderName_(iConfig.getUntrackedParameter<std::string>("HistogramFolderName","SiStrip/ReadoutView/FedMedians")),
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
    debugStream << "[SiStripFEDMediansPlugin]Configuration for SiStripFEDMediansPlugin: " << std::endl
                << "[SiStripFEDMediansPlugin]\tRawDataTag: " << rawDataTag_ << std::endl
                << "[SiStripFEDMediansPlugin]\tHistogramFolderName: " << folderName_ << std::endl
                << "[SiStripFEDMediansPlugin]\tPrintDebugMessages? " << (printDebug_ ? "yes" : "no") << std::endl
                << "[SiStripFEDMediansPlugin]\tWriteDQMStore? " << (writeDQMStore_ ? "yes" : "no") << std::endl;
    if (writeDQMStore_) debugStream << "[SiStripFEDMediansPlugin]\tDQMStoreFileName: " << dqmStoreFileName_ << std::endl;
  }
  
  if (printDebug_) {
    LogTrace("SiStripMonitorHardware") << debugStream.str();
  }

  const std::string psetName = "TimeHistogramConfig";
  if (iConfig.exists(psetName)) {
    const edm::ParameterSet& pset = iConfig.getUntrackedParameter<edm::ParameterSet>(psetName);
    timeConfig_.enabled = (pset.exists("Enabled") ? pset.getUntrackedParameter<bool>("Enabled") : true);
    if (timeConfig_.enabled) {
      timeConfig_.nBins = (pset.exists("NBins") ? pset.getUntrackedParameter<unsigned int>("NBins") : 600);
      timeConfig_.min = (pset.exists("Min") ? pset.getUntrackedParameter<double>("Min") : 0);
      timeConfig_.max = (pset.exists("Max") ? pset.getUntrackedParameter<double>("Max") : 40000);
    }
  }
}

SiStripFEDMediansPlugin::~SiStripFEDMediansPlugin()
{
}


//
// Member functions
//

// ------------ method called to for each event  ------------
void
SiStripFEDMediansPlugin::analyze(const edm::Event& iEvent, 
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

    bookFEDHistograms(fedId);


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
	    else if (shot->apvNumber()%2 == 1) {
	      lShotMedian.second = shot->median();
	      break;
	    }
	    //shot->nStrips()
	  }
	  //isFirst = false;
	}
      }

      //if (!isShot) continue;

      if (firstEvent){
	if (lSubDet == 3) {
	  TIBDetId lId(lDetId);
	  std::cout << "TIB layer " << lId.layer()  << ", fedID " << fedId << ", channel " << iCh << std::endl;
	}
      }
      
      if (isShot) {
	shotMedianAPV0_->Fill(lShotMedian.first);
	shotMedianAPV1_->Fill(lShotMedian.second);
      }

      std::ostringstream lMode;
      lMode << buffer->readoutMode();
  
      const sistrip::FEDChannel & lChannel = buffer->channel(iCh);

      std::pair<uint16_t,uint16_t> medians = fillMedians(lMode.str(),lChannel,iCh);

      if (printDebug_ > 1) {
	if (lChannel.length() > 7) {
	  infoStream << "Medians for channel #" << iCh << " (length " << lChannel.length() << "): " << medians.first << ", " << medians.second << std::endl;
	}
      }

      //if some clusters are found:
      if (lChannel.length() > 7) {
	medianAPV0_->Fill(medians.first);
	medianAPV1_->Fill(medians.second);
	medianAPV1vsAPV0_->Fill(medians.first,medians.second);
	medianAPV1minusAPV0_->Fill(medians.second-medians.first);
	if (isShot) {
	  diffMedianminusShotMedianAPV1_->Fill(medians.second-medians.first-lShotMedian.second);
	  medianAPV0minusShot0_->Fill(medians.first-(lShotMedian.first+128));
	  medianAPV1minusShot1_->Fill(medians.second-(lShotMedian.second+128));
	}
	medianAPV1minusAPV0perFED_[fedId]->Fill(medians.second-medians.first);
	medianAPV1minusAPV0vsChperFED_[fedId]->Fill(iCh,medians.second-medians.first);
	medianAPV1minusAPV0vsTimeperFED_[fedId]->Fill(iEvent.id().event(),medians.second-medians.first);
	medianAPV1vsTimeperFED_[fedId]->Fill(iEvent.id().event(),medians.second);
	medianAPV0vsTimeperFED_[fedId]->Fill(iEvent.id().event(),medians.first);
      }

    }//loop on channels
    
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
SiStripFEDMediansPlugin::beginJob(const edm::EventSetup&)
{
  //get DQM store
  dqm_ = &(*edm::Service<DQMStore>());
  dqm_->setCurrentFolder(folderName_);

  shotMedianAPV0_ = dqm_->book1D("shotMedianAPV0","median shot APV0",100,-50,50);
  shotMedianAPV1_ = dqm_->book1D("shotMedianAPV1","median shot APV1",100,-50,50);
  medianAPV0_ = dqm_->book1D("medianAPV0","median APV0",200,0,200);
  medianAPV1_ = dqm_->book1D("medianAPV1","median APV1",200,0,200);
  medianAPV1vsAPV0_ = dqm_->book2D("medianAPV1vsAPV0","median APV1 vs APV0",200,0,200,200,0,200);
  medianAPV1minusAPV0_ = dqm_->book1D("medianAPV1minusAPV0","median APV1 - median APV0",400,-200,200);
  diffMedianminusShotMedianAPV1_ = dqm_->book1D("diffMedianminusShotMedianAPV1","(median APV1 - median APV0)-shot median APV1",500,-50,50);
  medianAPV0minusShot0_ = dqm_->book1D("medianAPV0minusShot0","median APV0 - (median shot APV0+128)",100,-50,50);
  medianAPV1minusShot1_ = dqm_->book1D("medianAPV1minusShot1","median APV1 - (median shot APV1+128)",100,-50,50);

}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripFEDMediansPlugin::endJob()
{
  if (writeDQMStore_) dqm_->save(dqmStoreFileName_);
}

void SiStripFEDMediansPlugin::updateCabling(const edm::EventSetup& eventSetup)
{
  uint32_t currentCacheId = eventSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (cablingCacheId_ != currentCacheId) {
    edm::ESHandle<SiStripFedCabling> cablingHandle;
    eventSetup.get<SiStripFedCablingRcd>().get(cablingHandle);
    cabling_ = cablingHandle.product();
    cablingCacheId_ = currentCacheId;
  }
}


std::pair<uint16_t,uint16_t> SiStripFEDMediansPlugin::fillMedians(const std::string & aMode,
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
  uint16_t lMedian0 = lWord6 + (lWord7 << 8);
  uint16_t lMedian1 = lWord4 + (lWord5 << 8);

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



void SiStripFEDMediansPlugin::bookFEDHistograms(unsigned int fedId)
{

  std::ostringstream lTitle,lTitleCh,lTitleTime,lTitleTime0,lTitleTime1;
  lTitle << "medianAPV1minusAPV0_" << fedId ;
  lTitleCh << "medianAPV1minusAPV0vsCh_" << fedId ;
  lTitleTime << "medianAPV1minusAPV0vsTime_" << fedId ;
  lTitleTime0 << "medianAPV0vsTime_" << fedId ;
  lTitleTime1 << "medianAPV1vsTime_" << fedId ;
  medianAPV1minusAPV0perFED_[fedId] = dqm_->book1D(lTitle.str().c_str(),"median APV1 - median APV0",400,-200,200);
  medianAPV1minusAPV0vsChperFED_[fedId] = dqm_->bookProfile(lTitleCh.str().c_str(),"median APV1 - median APV0",96,0,96,-200,200);
  medianAPV1minusAPV0vsTimeperFED_[fedId] = dqm_->bookProfile(lTitleTime.str().c_str(),"median APV1 - median APV0",timeConfig_.nBins,timeConfig_.min,timeConfig_.max,-200,200);
  medianAPV0vsTimeperFED_[fedId] = dqm_->bookProfile(lTitleTime0.str().c_str(),"median APV0",timeConfig_.nBins,timeConfig_.min,timeConfig_.max,0,200);
  medianAPV1vsTimeperFED_[fedId] = dqm_->bookProfile(lTitleTime1.str().c_str(),"median APV1",timeConfig_.nBins,timeConfig_.min,timeConfig_.max,0,200);


}




#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDMediansPlugin);
