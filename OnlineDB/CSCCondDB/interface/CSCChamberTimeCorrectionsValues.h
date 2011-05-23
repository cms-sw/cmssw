#ifndef _CSCCCHAMBERTIMECORRECTIONSVALUES_H
#define _CSCCCHAMBERTIMECORRECTIONSVALUES_H

#include <memory>
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
#include "CondFormats/CSCObjects/interface/CSCChamberTimeCorrections.h"
#include "CondFormats/DataRecord/interface/CSCChamberTimeCorrectionsRcd.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberTimeCorrectionsValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCCableRead.h"

#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

class CSCChamberTimeCorrectionsValues: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCChamberTimeCorrectionsValues(const edm::ParameterSet&);
  ~CSCChamberTimeCorrectionsValues();
  
  inline static CSCChamberTimeCorrections * prefill(bool isMC, float ME11offset, float nonME11offset);

  typedef const  CSCChamberTimeCorrections * ReturnType;
  
  ReturnType produceChamberTimeCorrections(const CSCChamberTimeCorrectionsRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCChamberTimeCorrections *chamberObj ;

  //Flag for determining if this is for setting MC or data corrections
  bool isForMC;
  float ME11offsetMC;
  float ME11offsetData;
  float nonME11offsetMC;
  float nonME11offsetData;

};

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCChamberTimeCorrections *  CSCChamberTimeCorrectionsValues::prefill(bool isMC, float ME11offset, float nonME11offset)
{
  if (isMC)
    printf("\n Generating fake DB constants for MC\n");
  else 
    printf("\n Getting chamber corrections from the cable data base and possibly other files \n");

  const int FACTOR=100;
  const int MAX_SIZE = 540;
  //const int MAX_SHORT= 32767;

  CSCChamberTimeCorrections * chamberObj = new CSCChamberTimeCorrections();

  int i; //i - chamber index.
  int count=0;
  std::string chamber_label, cfeb_rev, alct_rev;
  float cfeb_length=0, alct_length=0, cfeb_tmb_skew_delay=0, cfeb_timing_corr=0;

  // Only the first 481 chambers have interesting cable lengths at present
  // The rest of the chambers will be filled with zeros
  chamberObj->factor_precision = FACTOR;

  chamberObj->chamberCorrections.resize(MAX_SIZE);
  // fill the database with dummy values
  for(i=1;i<=MAX_SIZE;++i){
    chamberObj->chamberCorrections[i-1].cfeb_length=0;
    chamberObj->chamberCorrections[i-1].cfeb_rev='X';
    chamberObj->chamberCorrections[i-1].alct_length=0;
    chamberObj->chamberCorrections[i-1].alct_rev='X';
    chamberObj->chamberCorrections[i-1].cfeb_tmb_skew_delay=0;
    chamberObj->chamberCorrections[i-1].cfeb_timing_corr=0;
    chamberObj->chamberCorrections[i-1].cfeb_cable_delay=0;
  }


  // for MC there will be one value for ME1/1 and one for all other chambers
  if (isMC){
    for(i=1;i<=MAX_SIZE;++i){
      if (i<= 36 || (i>= 235 && i<=270))
	chamberObj->chamberCorrections[i-1].cfeb_timing_corr=(short int)(-1*ME11offset*FACTOR+0.5*(-1*ME11offset>=0)-0.5*(-1*ME11offset<0));
      else
	chamberObj->chamberCorrections[i-1].cfeb_timing_corr=(short int)(-1*nonME11offset*FACTOR+0.5*(-1*nonME11offset>=0)-0.5*(-1*nonME11offset<0));
    }

    return chamberObj;
  }

  // ***************************************************************************
  // Everything below this point is for setting the chamber corrections for data
  // ***************************************************************************

  csccableread *cable = new csccableread ();
  for(i=1;i<=MAX_SIZE;++i){
    // for data we will read in from Igor's database
    cable->cable_read(i, &chamber_label, &cfeb_length, &cfeb_rev, &alct_length,
		      &alct_rev, &cfeb_tmb_skew_delay, &cfeb_timing_corr);
    // If the read of the cable database is useful (if there is information for the chamber there)
    // re-enter the information the cable object
    if(!chamber_label.empty() && !(cfeb_length==0)){
      chamberObj->chamberCorrections[i-1].cfeb_length=(short int)(cfeb_length*FACTOR+0.5);
      chamberObj->chamberCorrections[i-1].cfeb_rev=cfeb_rev[0];
      chamberObj->chamberCorrections[i-1].alct_length=(short int)(alct_length*FACTOR+0.5);
      chamberObj->chamberCorrections[i-1].alct_rev=alct_rev[0];
      chamberObj->chamberCorrections[i-1].cfeb_tmb_skew_delay=(short int)(cfeb_tmb_skew_delay*FACTOR+0.5);
      chamberObj->chamberCorrections[i-1].cfeb_timing_corr=(short int)(cfeb_timing_corr*FACTOR+0.5);
      chamberObj->chamberCorrections[i-1].cfeb_cable_delay=0;
    }
    count=count+1;
  }

  //Read in the changes you want to make in the extra chamber variable cfeb_timing_corr
  FILE *fin = fopen("/afs/cern.ch/user/d/deisher/public/TimingCorrections2009/ttcrx_delay_effects_23April_2010.txt","r");
  int chamber;
  float corr;
  while (!feof(fin)){
    //note space at end of format string to convert last \n
    int check = fscanf(fin,"%d %f \n",&chamber,&corr);
    if (check != 2){
      printf("cfeb timing corr file has an unexpected format \n");
      assert(0);  
    }
    //printf("chamber %d corr %f \n",chamber,corr);
    chamberObj->chamberCorrections[chamber-1].cfeb_timing_corr= (short int)(corr*FACTOR+0.5*(corr>=0)-0.5*(corr<0));
  }
  fclose(fin);  
  
  // Remove the offsets inherent to ME11 and non ME11 chambers
  for(i=1;i<=MAX_SIZE;++i){
    float temp= float(chamberObj->chamberCorrections[i-1].cfeb_timing_corr)/FACTOR;
    if (i<= 36 || (i>= 235 && i<=270))
      chamberObj->chamberCorrections[i-1].cfeb_timing_corr=(short int)((temp-1*ME11offset)*FACTOR+0.5*(temp>=ME11offset)-0.5*(temp<ME11offset));
    else
      chamberObj->chamberCorrections[i-1].cfeb_timing_corr=(short int)((temp-1*nonME11offset)*FACTOR+0.5*(temp>=nonME11offset)-0.5*(temp<nonME11offset));
  }

  //Read in the cfeb_cable_delay values (0 or 1) and don't use a precision correction factor 
  FILE *fdelay = fopen("/afs/cern.ch/user/d/deisher/public/TimingCorrections2009/cfeb_cable_delay_20100423_both.txt","r");
  char label[1024];
  int delay;
  CSCIndexer indexer;
  while (!feof(fdelay)){
    //note space at end of format string to convert last \n
    int check = fscanf(fdelay,"%1024s %d \n",label,&delay);
    if (check != 2){
      printf("cfeb cable delay file has an unexpected format \n");
      assert(0);  
    }
    int chamberSerial = 0;
    int c_endcap = (label[2]=='+' ? 1 : 2);
    int c_station = atoi(&label[3]);
    int c_ring = atoi(&label[5]);
    if (c_station ==1 && c_ring ==4)
      c_ring =1;
    int c_chamber = (label[7]=='0' ? atoi(&label[8]) : atoi(&label[7]));
    chamberSerial = indexer.chamberIndex(c_endcap, c_station, c_ring, c_chamber);
    //printf("chamberLabel %s (%d %d %d %d) chamberSerial %d delay %d \n",label,c_endcap,c_station, c_ring, c_chamber, chamberSerial,delay);
    chamberObj->chamberCorrections[chamberSerial-1].cfeb_cable_delay= (short int)delay;
  }
  fclose(fdelay);  
  
  return chamberObj;
}


#endif
