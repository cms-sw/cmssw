// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorHardware
// Class:      SiStripFEDMediansPlugin
// 
/**\class SiStripFEDMediansPlugin SiStripFEDMedians.cc DQM/SiStripMonitorHardware/plugins/SiStripFEDMedians.cc

 Description: DQM source application to produce data integrety histograms for SiStrip data
*/
//
// Original Author:  Nicholas Cripps
//         Created:  2008/09/16
// $Id: SiStripFEDMedians.cc,v 1.26 2009/07/09 14:34:53 amagnan Exp $
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
  MonitorElement *medianAPV1vsAPV0_;


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
    cablingCacheId_(0)
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
  //update cabling
  updateCabling(iSetup);
  
  //get raw data
  edm::Handle<FEDRawDataCollection> rawDataCollectionHandle;
  iEvent.getByLabel(rawDataTag_,rawDataCollectionHandle);
  const FEDRawDataCollection& rawDataCollection = *rawDataCollectionHandle;
  
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

    if (printDebug_ > 1) {
      std::ostringstream debugStream;
      debugStream << " --- Processing FED " << fedId << std::endl;
      edm::LogInfo("SiStripMonitorHardware") << debugStream.str();
    }
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
    
    //need to construct full object to go any further
    std::auto_ptr<const sistrip::FEDBuffer> buffer;
    buffer.reset(new sistrip::FEDBuffer(fedData.data(),fedData.size(),true));

    for (unsigned int iCh = 0; 
	 iCh < sistrip::FEDCH_PER_FED; 
	 iCh++) {//loop on channels
      bool connected = (cabling_->connection(fedId,iCh)).isConnected();
      if (!connected) continue;

      bool isBadChan = false;
      for (unsigned int badCh(0); badCh<lFedErrors.getBadChannels().size(); badCh++) {
	if (lFedErrors.getBadChannels().at(badCh).first == iCh) {
	  isBadChan = true;
	  break;
	}
      }

      if (isBadChan) continue;

      std::ostringstream lMode;
      lMode << buffer->readoutMode();
  
      std::pair<uint16_t,uint16_t> medians = fillMedians(lMode.str(),buffer->channel(iCh),iCh);

      if (printDebug_ > 1) {
	std::ostringstream debugStream;
	debugStream << "Medians for channel #" << iCh << ": " << medians.first << ", " << medians.second << std::endl;
	edm::LogInfo("SiStripMonitorHardware") << debugStream.str();
      }

      medianAPV0_->Fill(medians.first);
      medianAPV1_->Fill(medians.second);

      medianAPV1vsAPV0_->Fill(medians.first,medians.second);

    }//loop on channels

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

}//analyze method

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripFEDMediansPlugin::beginJob(const edm::EventSetup&)
{
  //get DQM store
  dqm_ = &(*edm::Service<DQMStore>());
  dqm_->setCurrentFolder(folderName_);

  medianAPV0_ = dqm_->book1D("medianAPV0","median APV0",100,110,150);
  medianAPV1_ = dqm_->book1D("medianAPV1","median APV1",100,110,150);
  medianAPV1vsAPV0_ = dqm_->book2D("medianAPV1vsAPV0","median APV1 vs APV0",100,110,150,100,110,150);

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
  if (printDebug_ > 1) std::cout << "Readout mode = " << aMode << std::endl;
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
  uint8_t lWord3 = lData[(aChannel.offset()+2)^7];
  uint8_t lWord4 = lData[(aChannel.offset()+3)^7];
  uint8_t lWord5 = lData[(aChannel.offset()+4)^7] & 0x03;
  uint8_t lWord6 = lData[(aChannel.offset()+5)^7];
  uint8_t lWord7 = lData[(aChannel.offset()+6)^7] & 0x03;

  uint16_t lLength  = lWord1 + (lWord2 << 8);
  uint16_t lMedian0 = lWord6 + (lWord7 << 8);
  uint16_t lMedian1 = lWord4 + (lWord5 << 8);

  if (lLength != aChannel.length()) {
    std::cout << "Channel #" << aIndex << " offset: " << aChannel.offset() << ", offset^7 = " << (aChannel.offset()^7) << std::endl;
    std::cout << "My length = " << lLength << ", Nicks's length = " << aChannel.length() << std::endl;
  }

  return std::pair<uint16_t,uint16_t>(lMedian0,lMedian1);
}

//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripFEDMediansPlugin);
