#ifndef CNASUB_ANALYZER_H
#define CNASUB_ANALYZER_H

// -*- C++ -*-
//
// Package:    EcalCorrelatedNoiseAnalysisModules
// Class:      EcalCorrelatedNoisePedestalRunAnalyzer
// 
/**\class EcalCorrelatedNoisePedestalRunAnalyzer EcalCorrelatedNoisePedestalRunAnalyzer.cc CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/src/EcalCorrelatedNoisePedestalRunAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Bernard Fabbro
//         Created:  Fri Jun  2 10:27:01 CEST 2006
// $Id$
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <time.h>
#include "Riostream.h"

// ROOT include files
#include "TObject.h"
#include "TSystem.h"
#include "TString.h"
#include "TVectorD.h"
#include "TTreeIndex.h"

// CMSSW include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"

// user include files
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaRunEB.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBNumbering.h"

//
// class declaration
//

class EcalCorrelatedNoisePedestalRunAnalyzer : public edm::EDAnalyzer {

 public:
  
  enum { kChannels = 1700, kGains = 3, kFirstGainId = 1 };
  
  
  explicit EcalCorrelatedNoisePedestalRunAnalyzer(const edm::ParameterSet&);
  ~EcalCorrelatedNoisePedestalRunAnalyzer();  
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:

  Int_t        fgMaxCar;   // Max nb of caracters for char*

  // ----------member data ---------------------------
  unsigned int verbosity_;
  Int_t nChannels_;
  Int_t iEvent_; // should be removed when we can access class EventID
  string eventHeaderProducer_;

  Int_t   fEvtNumber;

  Int_t   fFalseBurst1;
  Int_t   fFalseBurst2;
  Int_t   fFalseBurst3;

  TString fAnalysisName;
  Int_t   fRunNumber;
  Int_t   fSuperModule;
  Int_t   fFirstTakenEvt;   
  Int_t   fNbOfTakenEvents; // 150 = nb of events in burst for a given gain  => a aller chercher avec TCnaParameters 
  Int_t   fNentries;

  time_t   fTimeBurst1_1;
  time_t   fTimeBurst1_2;  
  TString  fDateBurst1_1;
  TString  fDateBurst1_2;

  time_t   fTimeBurst2_1;
  time_t   fTimeBurst2_2;  
  TString  fDateBurst2_1;
  TString  fDateBurst2_2;

  time_t   fTimeBurst3_1;
  time_t   fTimeBurst3_2;  
  TString  fDateBurst3_1;
  TString  fDateBurst3_2;


  TString     fTTBELL;

  TCnaRunEB*     fMyCnaRunEBBurst1;
  TCnaRunEB*     fMyCnaRunEBBurst2;
  TCnaRunEB*     fMyCnaRunEBBurst3;

  TEBNumbering*  fMyEBNumbering; 

};

#endif
