#ifndef _CSCCHIPSPEEDCORRECTIONDBCONDITIONS_H
#define _CSCCHIPSPEEDCORRECTIONDBCONDITIONS_H

#include <memory>
#include <cmath>
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "CondFormats/CSCObjects/interface/CSCDChipSpeedCorrection.h"
#include "CondFormats/DataRecord/interface/CSCDBChipSpeedCorrectionRcd.h"

class CSCChipSpeedCorrectionDBConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCChipSpeedCorrectionDBConditions(const edm::ParameterSet&);
  ~CSCChipSpeedCorrectionDBConditions();
  
  inline static CSCDBChipSpeedCorrection * prefillDBChipSpeedCorrection();

  typedef const  CSCDBChipSpeedCorrection * ReturnType;
  
  ReturnType produceDBChipSpeedCorrection(const CSCDBChipSpeedCorrectionRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCDBChipSpeedCorrection *cndbChipCorr ;

};

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "CondFormats/CSCObjects/interface/CSCDBChipSpeedCorrection.h"
#include "CondFormats/DataRecord/interface/CSCDBChipSpeedCorrectionRcd.h"

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCDBChipSpeedCorrection * CSCChipSpeedCorrectionDBConditions::prefillDBChipSpeedCorrection()  
{
  const int CHIP_FACTOR=10;
  const int MAX_SIZE = 252288;
  //const int MAX_SIZE = 273024; //for extra ME1a unganged case
  const int MAX_SHORT= 32767;
  CSCDBChipSpeedCorrection * cndbChipCorr = new CSCDBChipSpeedCorrection();


     
  /*
   return cndbChipCorr;
  */
}

#endif
