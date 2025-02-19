#ifndef _CSCFAKEDBPEDESTALS_H
#define _CSCFAKEDBPEDESTALS_H

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

#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <boost/shared_ptr.hpp>

class CSCFakeDBPedestals: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFakeDBPedestals(const edm::ParameterSet&);
      ~CSCFakeDBPedestals();
      
       inline static CSCDBPedestals * prefillDBPedestals();

      typedef  boost::shared_ptr<CSCDBPedestals> Pointer;

      Pointer produceDBPedestals(const CSCDBPedestalsRcd&);

   private:
      // ----------member data ---------------------------
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );

      Pointer cndbPedestals ;   
};

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCDBPedestals *  CSCFakeDBPedestals::prefillDBPedestals()
{
  int seed;
  float meanped,meanrms;
  const int MAX_SIZE = 217728; //or 252288 for ME4/2 chambers
  const int PED_FACTOR=10;
  const int RMS_FACTOR=1000;
 
  CSCDBPedestals * cndbpedestals = new CSCDBPedestals();
  cndbpedestals->pedestals.resize(MAX_SIZE);

  seed = 10000;	
  srand(seed);
  meanped=600.0, meanrms=1.5;
  cndbpedestals->factor_ped = int (PED_FACTOR);
  cndbpedestals->factor_rms = int (RMS_FACTOR);
 
  for(int i=0; i<MAX_SIZE;i++){
    cndbpedestals->pedestals[i].ped=(short int) (((double)rand()/((double)(RAND_MAX)+(double)(1)))*100+meanped*PED_FACTOR+0.5);
    cndbpedestals->pedestals[i].rms= (short int) (((double)rand()/((double)(RAND_MAX)+(double)(1)))+meanrms*RMS_FACTOR+0.5);
  }
  return cndbpedestals;
}  

#endif
