#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/EcalPreshowerMonitorModule/interface/ESFEDIntegrityTask.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/src/fed_header.h"
#include "DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h"
#include "DataFormats/EcalRawData/interface/ESKCHIPBlock.h"

using namespace cms;
using namespace edm;
using namespace std;

ESFEDIntegrityTask::ESFEDIntegrityTask(const ParameterSet& ps) {

  prefixME_      = ps.getUntrackedParameter<string>("prefixME", "");
  fedDirName_    = ps.getUntrackedParameter<string>("FEDDirName", "FEDIntegrity");
  debug_         = ps.getUntrackedParameter<bool>("debug", false);

  dccCollections_       = consumes<ESRawDataCollection>(ps.getParameter<InputTag>("ESDCCCollections"));
  FEDRawDataCollection_ = consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("FEDRawDataCollection"));

  meESFedsEntries_  = 0;
  meESFedsFatal_    = 0;
  meESFedsNonFatal_ = 0;

  ievt_ = 0;
  
}

void
ESFEDIntegrityTask::bookHistograms(DQMStore::IBooker& iBooker, Run const&, EventSetup const&)
{
  char histo[200];

  iBooker.setCurrentFolder(prefixME_ + "/" + fedDirName_);

  sprintf(histo, "FEDEntries");
  meESFedsEntries_ = iBooker.book1D(histo, histo, 56, 520, 576);

  sprintf(histo, "FEDFatal");
  meESFedsFatal_ = iBooker.book1D(histo, histo, 56, 520, 576);

  sprintf(histo, "FEDNonFatal");
  meESFedsNonFatal_ = iBooker.book1D(histo, histo, 56, 520, 576);
}

void ESFEDIntegrityTask::endJob(void){

  LogInfo("ESFEDIntegrityTask") << "analyzed " << ievt_ << " events";

}

void ESFEDIntegrityTask::analyze(const Event& e, const EventSetup& c){

  ievt_++;

  int gt_L1A = 0; 
  // int gt_OrbitNumber = 0, gt_BX = 0;
  int esDCC_L1A_MostFreqCounts = 0;
  int esDCC_BX_MostFreqCounts = 0;
  int esDCC_OrbitNumber_MostFreqCounts = 0;
  int gtFedDataSize = 0;

  Handle<ESRawDataCollection> dccs;
  Handle<FEDRawDataCollection> allFedRawData;

  if ( e.getByToken(FEDRawDataCollection_, allFedRawData) ) {

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
      //gt_OrbitNumber = e.orbitNumber();
      //gt_BX          = e.bunchCrossing();
    } else {

      map<int, int> esDCC_L1A_FreqMap;
      map<int, int> esDCC_BX_FreqMap;
      map<int, int> esDCC_OrbitNumber_FreqMap;

      if ( e.getByToken(dccCollections_, dccs) ) {
	
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
	    //gt_BX = esdcc.getBX();
	  } 
	  
	  if (esDCC_OrbitNumber_FreqMap[esdcc.getOrbitNumber()] > esDCC_OrbitNumber_MostFreqCounts) {
	    esDCC_OrbitNumber_MostFreqCounts = esDCC_OrbitNumber_FreqMap[esdcc.getOrbitNumber()];
	    //gt_OrbitNumber = esdcc.getOrbitNumber();
	  } 
	  
	}
      } else {
	LogWarning("ESFEDIntegrityTask") << "dccCollections not available";
      }

    }

  } else {
    LogWarning("ESFEDIntegrityTask") << "FEDRawDataCollection not available";
  }

  vector<int> fiberStatus;
  if ( e.getByToken(dccCollections_, dccs) ) {
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
