#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/EcalPreshowerMonitorModule/interface/ESFEDIntegrityTask.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/src/fed_header.h"
#include "DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h"
#include "DataFormats/EcalRawData/interface/ESKCHIPBlock.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

using namespace cms;
using namespace edm;
using namespace std;

ESFEDIntegrityTask::ESFEDIntegrityTask(const ParameterSet& ps) {

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_      = ps.getUntrackedParameter<string>("prefixME", "");
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);
  mergeRuns_     = ps.getUntrackedParameter<bool>("mergeRuns", false);
  debug_         = ps.getUntrackedParameter<bool>("debug", false);

  dccCollections_       = ps.getParameter<InputTag>("ESDCCCollections");
  kchipCollections_     = ps.getParameter<InputTag>("ESKChipCollections");
  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");

  meESFedsEntries_  = 0;
  meESFedsFatal_    = 0;
  meESFedsNonFatal_ = 0;
  
}

ESFEDIntegrityTask::~ESFEDIntegrityTask() {

}

void ESFEDIntegrityTask::beginJob(const EventSetup& c) {

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/FEDIntegrity");
    dqmStore_->rmdir(prefixME_ + "/FEDIntegrity");
  }

}

void ESFEDIntegrityTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void ESFEDIntegrityTask::endRun(const Run& r, const EventSetup& c) {

}

void ESFEDIntegrityTask::reset(void) {

  if ( meESFedsEntries_ )  meESFedsEntries_->Reset();
  if ( meESFedsFatal_ )    meESFedsFatal_->Reset();
  if ( meESFedsNonFatal_ ) meESFedsNonFatal_->Reset();

}

void ESFEDIntegrityTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/FEDIntegrity");

    sprintf(histo, "FEDEntries");
    meESFedsEntries_ = dqmStore_->book1D(histo, histo, 56, 520, 576);

    sprintf(histo, "FEDFatal");
    meESFedsFatal_ = dqmStore_->book1D(histo, histo, 56, 520, 576);

    sprintf(histo, "FEDNonFatal");
    meESFedsNonFatal_ = dqmStore_->book1D(histo, histo, 56, 520, 576);
  }

}

void ESFEDIntegrityTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/FEDIntegrity");

    if ( meESFedsEntries_ ) dqmStore_->removeElement( meESFedsEntries_->getName() );
    meESFedsEntries_ = 0;

    if ( meESFedsFatal_ ) dqmStore_->removeElement( meESFedsFatal_->getName() );
    meESFedsFatal_ = 0;

    if ( meESFedsNonFatal_ ) dqmStore_->removeElement( meESFedsNonFatal_->getName() );
    meESFedsNonFatal_ = 0;

  }

  init_ = false;

}

void ESFEDIntegrityTask::endJob(void){

  LogInfo("ESFEDIntegrityTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void ESFEDIntegrityTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  int gt_L1A = 0, gt_OrbitNumber = 0, gt_BX = 0;
  int esDCC_L1A_MostFreqCounts = 0;
  int esDCC_BX_MostFreqCounts = 0;
  int esDCC_OrbitNumber_MostFreqCounts = 0;
  int gtFedDataSize = 0;

  Handle<ESRawDataCollection> dccs;
  if ( e.getByLabel(dccCollections_, dccs) ) {
  } else {
    LogWarning("ESRawDataTask") << "Error! can't get ES raw data collection !" << std::endl;
  }

  edm::Handle<FEDRawDataCollection> allFedRawData;
  if ( e.getByLabel(FEDRawDataCollection_, allFedRawData) ) {

    // ES FEDs
    for (int esFED=520; esFED<=575; ++esFED) { 

      const FEDRawData& fedData = allFedRawData->FEDData(esFED);
      int length = fedData.size()/sizeof(uint64_t);
      
      if ( length > 0 ) 
	if ( meESFedsEntries_ ) meESFedsEntries_->Fill(esFED);
    }

    // GT FED data
    const FEDRawData& gtFedData = allFedRawData->FEDData(812);
     
    gtFedDataSize = gtFedData.size()/sizeof(uint64_t);
     
    if ( gtFedDataSize > 0 ) {
       
      FEDHeader header(gtFedData.data());
       
      gt_L1A         = header.lvl1ID();
      gt_OrbitNumber = e.orbitNumber();
      gt_BX          = e.bunchCrossing();
    } else {

      map<int, int> esDCC_L1A_FreqMap;
      map<int, int> esDCC_BX_FreqMap;
      map<int, int> esDCC_OrbitNumber_FreqMap;

      for (ESRawDataCollection::const_iterator dccItr = dccs->begin(); dccItr != dccs->end(); ++dccItr) {
	ESDCCHeaderBlock esdcc = (*dccItr);
	 
	esDCC_L1A_FreqMap[esdcc.getLV1()]++;
	esDCC_BX_FreqMap[esdcc.getBX()]++;
	esDCC_OrbitNumber_FreqMap[esdcc.getOrbitNumber()]++;
	 
	if (esDCC_L1A_FreqMap[esdcc.getLV1()] > esDCC_L1A_MostFreqCounts) {
	  esDCC_L1A_MostFreqCounts = esDCC_L1A_FreqMap[esdcc.getLV1()];
	  gt_L1A = esdcc.getLV1();
	} 

	if (esDCC_BX_FreqMap[esdcc.getBX()] > esDCC_BX_MostFreqCounts) {
	  esDCC_BX_MostFreqCounts = esDCC_BX_FreqMap[esdcc.getBX()];
	  gt_BX = esdcc.getBX();
	} 

	if (esDCC_OrbitNumber_FreqMap[esdcc.getOrbitNumber()] > esDCC_OrbitNumber_MostFreqCounts) {
	  esDCC_OrbitNumber_MostFreqCounts = esDCC_OrbitNumber_FreqMap[esdcc.getOrbitNumber()];
	  gt_OrbitNumber = esdcc.getOrbitNumber();
	} 

      }

    }

  } else {
    LogWarning("ESIntegrityTask") << FEDRawDataCollection_ << " not available";
  }

  vector<int> fiberStatus;
  for (ESRawDataCollection::const_iterator dccItr = dccs->begin(); dccItr != dccs->end(); ++dccItr) {
    ESDCCHeaderBlock dcc = (*dccItr);
    
    if (dcc.getDCCErrors() > 0) {
      
      if ( meESFedsFatal_ ) meESFedsFatal_->Fill(dcc.fedId());
      
    }	else {
      if (debug_) cout<<dcc.fedId()<<" "<<dcc.getOptoRX0()<<" "<<dcc.getOptoRX1()<<" "<<dcc.getOptoRX2()<<endl;
      fiberStatus = dcc.getFEChannelStatus();
      
      if (dcc.getOptoRX0() == 128) {
	meESFedsNonFatal_->Fill(dcc.fedId(), 1./3.);
      } else if (dcc.getOptoRX0() == 129) {
	for (unsigned int i=0; i<12; ++i) {
	  if (fiberStatus[i]==8 || fiberStatus[i]==10 || fiberStatus[i]==11 || fiberStatus[i]==12)
	    if ( meESFedsNonFatal_ ) meESFedsNonFatal_->Fill(dcc.fedId(), 1./12.);
	}
      }
      if (dcc.getOptoRX1() == 128) { 
	meESFedsNonFatal_->Fill(dcc.fedId(), 1./3.);
      }	else if (dcc.getOptoRX1() == 129) {
	for (unsigned int i=12; i<24; ++i) {
	  if (fiberStatus[i]==8 || fiberStatus[i]==10 || fiberStatus[i]==11 || fiberStatus[i]==12)
	    if ( meESFedsNonFatal_ ) meESFedsNonFatal_->Fill(dcc.fedId(), 1./12.);
	}
      }
      if (dcc.getOptoRX2() == 128) {
	meESFedsNonFatal_->Fill(dcc.fedId(), 1./3.);
      } else if (dcc.getOptoRX2() == 129){
	for (unsigned int i=24; i<36; ++i) {
	  if (fiberStatus[i]==8 || fiberStatus[i]==10 || fiberStatus[i]==11 || fiberStatus[i]==12)
	    if ( meESFedsNonFatal_ ) meESFedsNonFatal_->Fill(dcc.fedId(), 1./12.);
	}
      }
    }
    
    if (dcc.getLV1() != gt_L1A) meESFedsNonFatal_->Fill(dcc.fedId());
    //if (dcc.getBX() != gt_BX) meESFedsNonFatal_->Fill(dcc.fedId());
    //if (dcc.getOrbitNumber() != gt_OrbitNumber) meESFedsNonFatal_->Fill(dcc.fedId());
  }
  
  //for (ESLocalRawDataCollection::const_iterator kItr = kchips->begin(); kItr != kchips->end(); ++kItr) {

  //ESKCHIPBlock kchip = (*kItr);

  //Int_t nErr = 0;
  //if (kchip.getFlag1() > 0) nErr++; 
  //if (kchip.getFlag2() > 0) nErr++;
  //if (kchip.getBC() != kchip.getOptoBC()) nErr++;
  //if (kchip.getEC() != kchip.getOptoEC()) nErr++;
  //if (nErr>0) meESFedsNonFatal_->Fill(dcc.fedId());
  //}

}

DEFINE_FWK_MODULE(ESFEDIntegrityTask);
