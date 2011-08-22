#ifndef _CSCGASGAINCORRECTIONDBCONDITIONS_H
#define _CSCGASGAINCORRECTIONDBCONDITIONS_H

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
#include "CondFormats/CSCObjects/interface/CSCDBGasGainCorrection.h"
#include "CondFormats/DataRecord/interface/CSCDBGasGainCorrectionRcd.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

class CSCGasGainCorrectionDBConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCGasGainCorrectionDBConditions(const edm::ParameterSet&);
  ~CSCGasGainCorrectionDBConditions();
  
  inline static CSCDBGasGainCorrection * prefillDBGasGainCorrection(bool isForMC, std::string dataCorrFileName);

  typedef const  CSCDBGasGainCorrection * ReturnType;
  
  ReturnType produceDBGasGainCorrection(const CSCDBGasGainCorrectionRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCDBGasGainCorrection *cndbGasGainCorr ;

  //Flag for determining if this is for setting MC or data corrections
  bool isForMC;
  //File for reading 55944 gas gain corrections.  MC will be fake;
  std::string dataCorrFileName;

};

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "CondFormats/CSCObjects/interface/CSCDBGasGainCorrection.h"
#include "CondFormats/DataRecord/interface/CSCDBGasGainCorrectionRcd.h"

#include<fstream>
#include<vector>
#include<iostream>



// to workaround plugin library
inline CSCDBGasGainCorrection * CSCGasGainCorrectionDBConditions::prefillDBGasGainCorrection(bool isMC, std::string filename)  
{
  if (isMC)
    printf("\n Generating fake DB constants for MC\n");
  else {
    printf("\n Reading gas gain corrections from file %s \n",filename.data());
  }
  
  CSCIndexer indexer;

  const int MAX_SIZE = 55944;

  CSCDBGasGainCorrection * cndbGasGainCorr = new CSCDBGasGainCorrection();

  CSCDBGasGainCorrection::GasGainContainer & itemvector = cndbGasGainCorr->gasGainCorr;
  itemvector.resize(MAX_SIZE);

  //Filling corrections for MC is very simple
  if (isMC){
    for (int i=0;i<MAX_SIZE;i++){
      itemvector[i].gainCorr = 1.;
    }
    return cndbGasGainCorr;
  }

  struct gain_info {
    int   gas_gain_index     ;
    int   endcap             ;
    int   station            ;
    int   ring               ;
    int   chamber            ;
    int   layer              ;
    int   hvsegment          ;
    int   cfeb               ;
    int   nentries           ;
    float mean               ;
    float truncated_mean     ;
    float gas_gain_correction;
  } gains[MAX_SIZE];


  for (int j=0; j<MAX_SIZE; j++) {
    gains[j].gas_gain_index     = -999;
    gains[j].endcap             = -999;
    gains[j].station            = -999;
    gains[j].ring               = -999;
    gains[j].chamber            = -999;
    gains[j].layer              = -999;
    gains[j].hvsegment          = -999;
    gains[j].cfeb               = -999;
    gains[j].nentries           = -999;
    gains[j].mean               = -999.;
    gains[j].truncated_mean     = -999.;
    gains[j].gas_gain_correction= -999.;
  }

  FILE *fin = fopen(filename.data(),"r");

  int linecounter = 0;  // set the line counter to the first serial number in the file....

  while (!feof(fin)){
    //note space at end of format string to convert last \n
    int check = fscanf(fin,"%d %d %d %d %d %d %d %d %d %f %f %f \n",
		       &gains[linecounter].gas_gain_index     ,
		       &gains[linecounter].endcap             ,
		       &gains[linecounter].station            ,
		       &gains[linecounter].ring               ,
		       &gains[linecounter].chamber            ,
		       &gains[linecounter].layer              ,
		       &gains[linecounter].hvsegment          ,
		       &gains[linecounter].cfeb               ,
		       &gains[linecounter].nentries           ,
		       &gains[linecounter].mean               ,
		       &gains[linecounter].truncated_mean     ,
		       &gains[linecounter].gas_gain_correction);

    if (check != 12){
      printf("The input file format is not as expected\n");
      assert(0);
    }

    linecounter++;

  }

  fclose(fin);      

  if (linecounter == MAX_SIZE) {
    std::cout << "Total number of gas gains read in = " << linecounter << std::endl;
  } else {
    std::cout << "ERROR:  Total number of gas-gains read in = " << linecounter 
	      << " while total number expected = " << MAX_SIZE << std::endl;
  }

  // Fill the chip corrections with values from the file
  for (int i=0;i<MAX_SIZE;i++){

    itemvector[i].gainCorr = 0.;

    if (gains[i].gas_gain_correction > 0.) {
      itemvector[i].gainCorr = gains[i].gas_gain_correction;
    } else {
      // if there is no value, this should be fixed...
      std::cout << "ERROR:  gas_gain_correction < 0 for index " << gains[i].gas_gain_index << std::endl;
    }
  }
 
  return cndbGasGainCorr;
}
 
#endif
