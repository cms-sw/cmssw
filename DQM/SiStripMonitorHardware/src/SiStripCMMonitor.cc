
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
//

#include <sstream>
#include <memory>
#include <list>
#include <algorithm>
#include <cassert>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
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
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripMonitorHardware/interface/FEDHistograms.hh"
#include "DQM/SiStripMonitorHardware/interface/FEDErrors.hh"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DQM/SiStripMonitorHardware/interface/CMHistograms.hh"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

//
// Class declaration
//

class SiStripCMMonitorPlugin : public DQMEDAnalyzer
{
 public:

  explicit SiStripCMMonitorPlugin(const edm::ParameterSet&);
  ~SiStripCMMonitorPlugin() override;
 private:

  struct Statistics {
    float Mean;
    float Rms;
    float Counter;
  };

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run& , const edm::EventSetup& ) override;

  //update the cabling if necessary
  void updateCabling(const edm::EventSetup& eventSetup);


  void fillMaps(uint32_t aDetId, unsigned short aChInModule, std::pair<uint16_t,uint16_t> aMedians);

  //tag of FEDRawData collection
  edm::InputTag rawDataTag_;
  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;
  //folder name for histograms in DQMStore
  std::string folderName_;
  //vector of fedIDs which will have detailed histograms made
  std::vector<unsigned int> fedIdVec_;
  //book detailed histograms even if they will be empty (for merging)
  bool fillAllDetailedHistograms_;
  //do histos vs time with time=event number. Default time = orbit number (s)
  bool fillWithEvtNum_;
  bool fillWithLocalEvtNum_;
  //print debug messages when problems are found: 1=error debug, 2=light debug, 3=full debug
  unsigned int printDebug_;
  //FED cabling
  uint32_t cablingCacheId_;
  const SiStripFedCabling* cabling_;

  //add parameter to save computing time if TkHistoMap are not filled
  bool doTkHistoMap_;

  CMHistograms cmHists_;

  std::map<unsigned int,Statistics> CommonModes_;
  std::map<unsigned int,Statistics> CommonModesAPV0minusAPV1_;

  std::pair<uint16_t,uint16_t> prevMedians_[FEDNumbering::MAXSiStripFEDID+1][sistrip::FEDCH_PER_FED];

  edm::EventNumber_t evt_;

};


//
// Constructors and destructor
//

SiStripCMMonitorPlugin::SiStripCMMonitorPlugin(const edm::ParameterSet& iConfig)
  : rawDataTag_(iConfig.getUntrackedParameter<edm::InputTag>("RawDataTag",edm::InputTag("source",""))),
    folderName_(iConfig.getUntrackedParameter<std::string>("HistogramFolderName","SiStrip/ReadoutView/CMMonitoring")),
    fedIdVec_(iConfig.getUntrackedParameter<std::vector<unsigned int> >("FedIdVec")),
    fillAllDetailedHistograms_(iConfig.getUntrackedParameter<bool>("FillAllDetailedHistograms",false)),
    fillWithEvtNum_(iConfig.getUntrackedParameter<bool>("FillWithEventNumber",false)),
    fillWithLocalEvtNum_(iConfig.getUntrackedParameter<bool>("FillWithLocalEventNumber",false)),
    printDebug_(iConfig.getUntrackedParameter<unsigned int>("PrintDebugMessages",1)),
    cablingCacheId_(0)
    
{
  rawDataToken_ = consumes<FEDRawDataCollection>(rawDataTag_);
  //print config to debug log
  std::ostringstream debugStream;
  if (printDebug_>1) {
    debugStream << "[SiStripCMMonitorPlugin]Configuration for SiStripCMMonitorPlugin: " << std::endl
                << "[SiStripCMMonitorPlugin]\tRawDataTag: " << rawDataTag_ << std::endl
                << "[SiStripCMMonitorPlugin]\tHistogramFolderName: " << folderName_ << std::endl
                << "[SiStripCMMonitorPlugin]\tFillAllDetailedHistograms? " << (fillAllDetailedHistograms_ ? "yes" : "no") << std::endl
		<< "[SiStripCMMonitorPlugin]\tFillWithEventNumber?" << (fillWithEvtNum_ ? "yes" : "no") << std::endl
                << "[SiStripCMMonitorPlugin]\tPrintDebugMessages? " << (printDebug_ ? "yes" : "no") << std::endl;
  }
    
 std::ostringstream* pDebugStream = (printDebug_>1 ? &debugStream : nullptr);

 cmHists_.initialise(iConfig,pDebugStream);

 doTkHistoMap_ = cmHists_.tkHistoMapEnabled();

 CommonModes_.clear();
 CommonModesAPV0minusAPV1_.clear();

 for (unsigned int fedId(FEDNumbering::MINSiStripFEDID); fedId <= FEDNumbering::MAXSiStripFEDID; fedId++){
   for (unsigned int iCh(0); iCh<sistrip::FEDCH_PER_FED; iCh++){
     prevMedians_[fedId][iCh] = std::pair<uint16_t,uint16_t>(0,0);
   }
 }


 if (printDebug_)
   LogTrace("SiStripMonitorHardware") << debugStream.str();

 evt_ = 0;

}

SiStripCMMonitorPlugin::~SiStripCMMonitorPlugin()
{
}


//
// Member functions
//


void SiStripCMMonitorPlugin::bookHistograms(DQMStore::IBooker & ibooker , const edm::Run & run, const edm::EventSetup & eSetup)
{
  ibooker.setCurrentFolder(folderName_);

  edm::ESHandle<TkDetMap> tkDetMapHandle;
  eSetup.get<TrackerTopologyRcd>().get(tkDetMapHandle);
  const TkDetMap* tkDetMap = tkDetMapHandle.product();

  cmHists_.bookTopLevelHistograms(ibooker, tkDetMap);

  if (fillAllDetailedHistograms_) cmHists_.bookAllFEDHistograms(ibooker);
}

void SiStripCMMonitorPlugin::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) 
{

}

// ------------ method called to for each event  ------------
void
SiStripCMMonitorPlugin::analyze(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //static bool firstEvent = true;
  //static bool isBeingFilled = false;
  //update cabling
  updateCabling(iSetup);
  
  //get raw data
  edm::Handle<FEDRawDataCollection> rawDataCollectionHandle;
  iEvent.getByToken(rawDataToken_,rawDataCollectionHandle);
  const FEDRawDataCollection& rawDataCollection = *rawDataCollectionHandle;
  
  //FED errors
  FEDErrors lFedErrors;

  //loop over siStrip FED IDs
  for (unsigned int fedId = FEDNumbering::MINSiStripFEDID; 
       fedId <= FEDNumbering::MAXSiStripFEDID; 
       fedId++) {//loop over FED IDs
    const FEDRawData& fedData = rawDataCollection.FEDData(fedId);

    //create an object to fill all errors
    lFedErrors.initialiseFED(fedId,cabling_,tTopo);

    //Do detailed check
    //first check if data exists
    bool lDataExist = lFedErrors.checkDataPresent(fedData);
    if (!lDataExist) {
      continue;
    }

    std::unique_ptr<const sistrip::FEDBuffer> buffer;

    if (!lFedErrors.fillFatalFEDErrors(fedData,0)) {
      continue;
    }
    else {
      //need to construct full object to go any further
      buffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size(),true));
      bool channelLengthsOK = buffer->checkChannelLengthsMatchBufferLength();
      bool channelPacketCodesOK = buffer->checkChannelPacketCodes();
      bool feLengthsOK = buffer->checkFEUnitLengths();
      if ( !channelLengthsOK ||
	   !channelPacketCodesOK ||
	   !feLengthsOK ) {
	continue;
      }
    }

    std::ostringstream infoStream;

    
    if (printDebug_ > 1) {
      infoStream << " --- Processing FED #" << fedId << std::endl;
    }


    std::vector<CMHistograms::CMvalues> values;

    for (unsigned int iCh = 0; 
	 iCh < sistrip::FEDCH_PER_FED; 
	 iCh++) {//loop on channels

      const FedChannelConnection & lConnection = cabling_->fedConnection(fedId,iCh);
      bool connected = lConnection.isConnected();

      //std::cout << "FedID " << fedId << ", ch " << iCh << ", nAPVPairs " << lConnection.nApvPairs() << " apvPairNumber " << lConnection.apvPairNumber() << std::endl;

      if (!connected) {
	continue;
      }

      uint32_t lDetId = lConnection.detId();
      unsigned short nChInModule = lConnection.nApvPairs();

      if (!lDetId || lDetId == sistrip::invalid32_) continue;

      bool lFailUnpackerChannelCheck = !buffer->channelGood(iCh, true) && connected;

      if (lFailUnpackerChannelCheck) {
	continue;
      }
      

      //short lAPVPair = lConnection.apvPairNumber();
      //short lSubDet = DetId(lDetId).subdetId();

//       if (firstEvent){
// 	infoStream << "Subdet " << lSubDet << ", " ;
// 	if (lSubDet == 3) {
// 	  
// 	  infoStream << "TIB layer " << tTopo->tibLayer(lDetId)  << ", fedID " << fedId << ", channel " << iCh << std::endl;
// 	}
// 	else if (lSubDet == 4) {
// 	  
// 	  infoStream << "TID side " << tTopo->tibSide(lDetId)  << " wheel " << tTopo->tibWheel(lDetId) << ", ring " << tTopo->tibRing(lDetId) << ", fedID " << fedId << ", channel " << iCh << std::endl;
// 	}
// 	else if (lSubDet == 5) {
// 	  
// 	  infoStream << "TOB side " << tTopo->tibRod(lDetId)[0]  << " layer " << tTopo->tibLayer(lDetId) << ", rod " << tTopo->tibRodNumber(lDetId) << ", fedID " << fedId << ", channel " << iCh << std::endl;
// 	}
// 	else if (lSubDet == 6) {
// 	  
// 	  infoStream << "TEC side " << tTopo->tibSide(lDetId)  << " wheel " << tTopo->tibWheel(lDetId) << ", petal " << tTopo->tibPetalNumber(lDetId) << ", ring " << tTopo->tibRing(lDetId) << ", fedID " << fedId << ", channel " << iCh << std::endl;
// 	}
// 	isBeingFilled=true;
//       }

      std::ostringstream lMode;
      lMode << buffer->readoutMode();
      if (evt_ == 0 && printDebug_ > 1) 
	std::cout << "Readout mode: " << lMode.str() << std::endl;
      

      const sistrip::FEDChannel & lChannel = buffer->channel(iCh);
      std::pair<uint16_t,uint16_t> medians = std::pair<uint16_t,uint16_t>(0,0);

      if (lMode.str().find("Zero suppressed") != lMode.str().npos && lMode.str().find("lite") == lMode.str().npos) medians = std::pair<uint16_t,uint16_t>(lChannel.cmMedian(0),lChannel.cmMedian(1));
      
      CMHistograms::CMvalues lVal;
      lVal.ChannelID = iCh;
      lVal.Medians = std::pair<uint16_t,uint16_t>(medians.first,medians.second);
      lVal.PreviousMedians = prevMedians_[fedId][iCh];

//       if (medians.second-medians.first > 26){
// 	std::ostringstream info;
// 	if (medians.second-medians.first > 44) info << " --- Second bump: event " << iEvent.id().event() << ", FED/Channel " << fedId << "/" << iCh << ", delta=" << medians.second-medians.first << std::endl;
// 	else info << " --- First bump: event " << iEvent.id().event() << ", FED/Channel " << fedId << "/" << iCh << ", delta=" << medians.second-medians.first << std::endl;
// 	edm::LogVerbatim("SiStripMonitorHardware") << info.str();
//       }

      if (printDebug_ > 1) {
	if (lChannel.length() > 7) {
	  infoStream << "Medians for channel #" << iCh << " (length " << lChannel.length() << "): " << medians.first << ", " << medians.second << std::endl;
	}
      }

      values.push_back(lVal);

      //if (iEvent.id().event() > 1000)
      fillMaps(lDetId,nChInModule,medians);

      prevMedians_[fedId][iCh] = std::pair<uint16_t,uint16_t>(medians.first,medians.second);
      
    }//loop on channels
    
    float lTime = 0;
    if (fillWithEvtNum_) {
      // casting from unsigned long long to a float here
      // doing it explicitely
      lTime = static_cast<float>(iEvent.id().event());
    } else {
      if (fillWithLocalEvtNum_) {
        // casting from unsigned long long to a float here
        // doing it explicitely
        lTime = static_cast<float>(evt_);
      } else {
        lTime = iEvent.orbitNumber()/11223.;
      }
    }

    cmHists_.fillHistograms(values,lTime,fedId);

    //if (printDebug_ > 0 && isBeingFilled && firstEvent) edm::LogVerbatim("SiStripMonitorHardware") << infoStream.str();
 


  }//loop on FEDs


  //if (isBeingFilled) 
  //firstEvent = false;

  evt_++;

}//analyze method

// ------------ method called once each job just after ending the event loop  ------------
/* //to be moved to harvesting step
void 
SiStripCMMonitorPlugin::endJob()
{

  if (doTkHistoMap_) {//if TkHistoMap is enabled
    std::map<unsigned int,Statistics>::iterator fracIter;

    //int ele = 0;
    //int nBadChannels = 0;
    for (fracIter = CommonModes_.begin(); fracIter!=CommonModes_.end(); fracIter++){
      uint32_t detid = fracIter->first;
      //if ((fracIter->second).second != 0) {
      //std::cout << "------ ele #" << ele << ", Frac for detid #" << detid << " = " <<(fracIter->second).second << "/" << (fracIter->second).first << std::endl;
      //nBadChannels++;
      //}
      float mean = 0;
      float rms = 0;
      Statistics lStat = fracIter->second;
      if (lStat.Counter > 0) mean = lStat.Mean/lStat.Counter;
      if (lStat.Counter > 1) rms = sqrt(lStat.Rms/(lStat.Counter-1)-(mean*mean));
      cmHists_.fillTkHistoMap(cmHists_.tkHistoMapPointer(0),detid,mean);
      cmHists_.fillTkHistoMap(cmHists_.tkHistoMapPointer(1),detid,rms);

      if (printDebug_ > 1) {
	std::ostringstream message;
	message << "TkHistoMap CM: Detid " << detid << ", mean = " <<  mean << ", rms = " << rms << ", counter = " << lStat.Counter << std::endl;
	edm::LogVerbatim("SiStripMonitorHardware") << message.str();
      }

      //ele++;
    }

    for (fracIter = CommonModesAPV0minusAPV1_.begin(); fracIter!=CommonModesAPV0minusAPV1_.end(); fracIter++){
      uint32_t detid = fracIter->first;
      //if ((fracIter->second).second != 0) {
      //std::cout << "------ ele #" << ele << ", Frac for detid #" << detid << " = " <<(fracIter->second).second << "/" << (fracIter->second).first << std::endl;
      //nBadChannels++;
      //}
      float mean = 0;
      float rms = 0;
      Statistics lStat = fracIter->second;
      if (lStat.Counter > 0) mean = lStat.Mean/lStat.Counter;
      if (lStat.Counter > 1) rms = sqrt(lStat.Rms/(lStat.Counter-1)-(mean*mean));
      cmHists_.fillTkHistoMap(cmHists_.tkHistoMapPointer(2),detid,mean);
      cmHists_.fillTkHistoMap(cmHists_.tkHistoMapPointer(3),detid,rms);

      if (printDebug_ > 1) {
	std::ostringstream message;
	message << "TkHistoMap APV0minusAPV1: Detid " << detid << ", mean = " <<  mean << ", rms = " << rms << ", counter = " << lStat.Counter << std::endl;
	edm::LogVerbatim("SiStripMonitorHardware") << message.str();
      }

      //ele++;
    }

  }//if TkHistoMap is enabled

}
*/
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


void SiStripCMMonitorPlugin::fillMaps(uint32_t aDetId, unsigned short aChInModule, std::pair<uint16_t,uint16_t> aMedians)
{

  if (doTkHistoMap_){//if TkHistMap is enabled
    std::pair<std::map<unsigned int,Statistics>::iterator,bool> alreadyThere[2];

    Statistics lStat;
    lStat.Mean = (aMedians.first+aMedians.second)*1./(2*aChInModule);
    lStat.Rms = (aMedians.first+aMedians.second)*(aMedians.first+aMedians.second)*1./(4*aChInModule);
    lStat.Counter = 1./aChInModule;

    alreadyThere[0] = CommonModes_.insert(std::pair<unsigned int,Statistics>(aDetId,lStat));
    if (!alreadyThere[0].second) {
      ((alreadyThere[0].first)->second).Mean += (aMedians.first+aMedians.second)*1./(2*aChInModule);
      ((alreadyThere[0].first)->second).Rms += (aMedians.first+aMedians.second)*(aMedians.first+aMedians.second)*1./(4*aChInModule);
      ((alreadyThere[0].first)->second).Counter += 1./aChInModule;
    }

    lStat.Mean = (aMedians.first-aMedians.second)*1./aChInModule;
    lStat.Rms = (aMedians.first-aMedians.second)*(aMedians.first-aMedians.second)*1./aChInModule;
    lStat.Counter = 1./aChInModule;

    alreadyThere[1] = CommonModesAPV0minusAPV1_.insert(std::pair<unsigned int,Statistics>(aDetId,lStat));
    if (!alreadyThere[1].second) {
      ((alreadyThere[1].first)->second).Mean += (aMedians.first-aMedians.second)*1./aChInModule;
      ((alreadyThere[1].first)->second).Rms += (aMedians.first-aMedians.second)*(aMedians.first-aMedians.second)*1./aChInModule;
      ((alreadyThere[1].first)->second).Counter += 1./aChInModule;
    }

  }

}


//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripCMMonitorPlugin);
