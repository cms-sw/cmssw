#ifndef _CSCFAKEDBGAINS_H
#define _CSCFAKEDBGAINS_H

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

#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCFakeDBGains: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
   public:
      CSCFakeDBGains(const edm::ParameterSet&);
      ~CSCFakeDBGains() override;

      inline static CSCDBGains* prefillDBGains(); 

      typedef std::shared_ptr<CSCDBGains> Pointer;
      Pointer produceDBGains(const CSCDBGainsRcd&);

   private:
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & ) override;

    // member data
    Pointer cndbGains ;

};

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCDBGains* CSCFakeDBGains::prefillDBGains()
{
  int seed;
  float mean;
  const int MAX_SIZE = 217728; //or 252288 for ME4/2 chambers
  const int FACTOR=1000;
  
  CSCDBGains* cndbgains = new CSCDBGains();
  cndbgains->gains.resize(MAX_SIZE);

  seed = 10000; 
  srand(seed);
  mean=6.8;
  cndbgains->factor_gain = int (FACTOR);
  
  for(int i=0; i<MAX_SIZE;i++){
    cndbgains->gains[i].gain_slope= (short int) (((double)rand()/((double)(RAND_MAX)+(double)(1)))+mean*FACTOR+0.5);
  }
  return cndbgains;
}  

#endif
