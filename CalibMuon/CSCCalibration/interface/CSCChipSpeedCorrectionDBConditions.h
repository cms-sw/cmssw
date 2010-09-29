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
#include "CondFormats/CSCObjects/interface/CSCDBChipSpeedCorrection.h"
#include "CondFormats/DataRecord/interface/CSCDBChipSpeedCorrectionRcd.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

class CSCChipSpeedCorrectionDBConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCChipSpeedCorrectionDBConditions(const edm::ParameterSet&);
  ~CSCChipSpeedCorrectionDBConditions();
  
  inline static CSCDBChipSpeedCorrection * prefillDBChipSpeedCorrection(bool isForMC, std::string dataCorrFileName, float dataOffse);

  typedef const  CSCDBChipSpeedCorrection * ReturnType;
  
  ReturnType produceDBChipSpeedCorrection(const CSCDBChipSpeedCorrectionRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCDBChipSpeedCorrection *cndbChipCorr ;

  //Flag for determining if this is for setting MC or data corrections
  bool isForMC;
  //File for reading <= 15768 data chip corrections.  MC will be fake (only one value for every chip);
  std::string dataCorrFileName;
  float dataOffset;

};

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "CondFormats/CSCObjects/interface/CSCDBChipSpeedCorrection.h"
#include "CondFormats/DataRecord/interface/CSCDBChipSpeedCorrectionRcd.h"

#include<fstream>
#include<vector>
#include<iostream>



// to workaround plugin library
inline CSCDBChipSpeedCorrection * CSCChipSpeedCorrectionDBConditions::prefillDBChipSpeedCorrection(bool isMC, std::string filename,float dataOffset)  
{
  if (isMC)
    printf("\n Generating fake DB constants for MC\n");
  else {
    printf("\n Reading chip corrections from file %s \n",filename.data());
    printf("my data offset value is %f \n",dataOffset);
  }
  
  CSCIndexer indexer;

  const int CHIP_FACTOR=100;
  const int MAX_SIZE = 15768;
  const int MAX_SHORT= 32767;
  CSCDBChipSpeedCorrection * cndbChipCorr = new CSCDBChipSpeedCorrection();

  CSCDBChipSpeedCorrection::ChipSpeedContainer & itemvector = cndbChipCorr->chipSpeedCorr;
  itemvector.resize(MAX_SIZE);
  cndbChipCorr->factor_speedCorr= int (CHIP_FACTOR);

  //Fill chip corrections for MC is very simple
  if (isMC){
    for (int i=0;i<MAX_SIZE;i++){
      itemvector[i].speedCorr = 0;
    }
    return cndbChipCorr;
  }

  //Filling for data takes a little more time
  FILE *fin = fopen(filename.data(),"r");

  int serialChamber,endcap,station,ring,chamber,chip;
  float t,dt;
  int nPulses;

  std::vector<int> new_index_id;
  std::vector<float> new_chipPulse;

  while (!feof(fin)){
    //note space at end of format string to convert last \n
    // int check = fscanf(fin,"%d %d %f %f \n",&serialChamber,&chip,&t,&dt);
    int check = fscanf(fin,"%d %d %d %d %d %d %f %f %d \n",&serialChamber,&endcap,&station,&ring,&chamber,&chip,&t,&dt,&nPulses);
    if (check != 9){
      printf("The input file format is not as expected\n");
      assert(0);
    }
   
    int serialChamber_safecopy = serialChamber;
    // Now to map from S. Durkin's serialChamber index convention 
    // ME+1/1-ME+41 = 1  -234
    // ME+4/2       = 235-270*
    // ME-1/1-ME-41 = 271-504*
    // ME-4/2       = 505-540
    // To the convention used in /DataFormats/MuonDetId/interface/CSCIndexer.h
    // ME+1/1-ME+41 = 1  -234
    // ME+4/2       = 469-504*
    // ME-1/1-ME-41 = 235-468*
    // ME-4/2       = 505-540
    // Only the chambers marked * need to be remapped
    
    if (serialChamber >=235 && serialChamber <=270)
      serialChamber += 234;
    else { // not in ME+4\2
      if (serialChamber >=271 && serialChamber <=504)
	serialChamber -= 36;
    }

    CSCDetId chamberId = indexer.detIdFromChamberIndex(serialChamber);
    // Now to map from S. Durkin's chip index convention 0-29 (with 4,9,14,19,24,29 unused in ME1/3)
    // To the convention used in /DataFormats/MuonDetId/interface/CSCIndexer.h 
    // Layer 1-6 and chip 1-5 for all chambers except ME1/3 which is chip 1-4
    int layer = (chip)/5+1;
    CSCDetId cscId(chamberId.endcap(),chamberId.station(),chamberId.ring(),chamberId.chamber(),layer);
    // This should yield the same CSCDetId as decoding the chamber serial does
    // If not, some debugging is needed
    CSCDetId cscId_doubleCheck(endcap,station,ring,chamber,layer);
    if (cscId != cscId_doubleCheck){
      printf("Why doesn't chamberSerial %d map to e: %d s: %d r: %d c: %d ? \n",serialChamber_safecopy, endcap, station, ring, chamber);
      assert(0);
    }

    chip =(chip%5)+1;
    // The file produced by Stan starts from the geometrical strip number stored in the CSCStripDigi 
    // When the strip digis are built, the conversion from electronics channel to geometrical strip (reversing ME+1/1a and, more importantly, ME-1/1b) 
    // is done before the digi is filled (EventFilter/CSCRawToDigi/src/CSCCFEBData.cc).
    // Since we're filling an electronics channel database, I'll flip again ME-1/1b
    if (endcap == 2 && station ==1 && ring==1 && chip <5){
      chip = 5 - chip; //change 1-4 to 4-1
    }


    new_index_id.push_back(indexer.chipIndex(cscId,chip));
    new_chipPulse.push_back(t);
  }
  fclose(fin);    


  // Fill the chip corrections with zeros to start
  for (int i=0;i<MAX_SIZE;i++){
    itemvector[i].speedCorr = 0;
  }
 
  for (unsigned int i=0;i<new_index_id.size();i++){
    if((short int)  (fabs((dataOffset-new_chipPulse[i])*CHIP_FACTOR+0.5))<MAX_SHORT) 
      itemvector[new_index_id[i]-1].speedCorr = (short int) ((dataOffset-new_chipPulse[i])*CHIP_FACTOR+0.5*(dataOffset>=new_chipPulse[i])-0.5*(dataOffset<new_chipPulse[i]));
    //printf("i= %d \t new index id = %d \t corr = %f \n",i,new_index_id[i], new_chipPulse[i]);
  }

  return cndbChipCorr;
}

#endif
