#ifndef _CSCGAINSDBCONDITIONS_H
#define _CSCGAINSDBCONDITIONS_H

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
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"

class CSCGainsDBConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCGainsDBConditions(const edm::ParameterSet&);
  ~CSCGainsDBConditions();
  

  inline static CSCDBGains *  prefillDBGains();

  typedef const  CSCDBGains * ReturnType;
  
  ReturnType produceDBGains(const CSCDBGainsRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCDBGains *cndbGains ;

};

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCDBGains *  CSCGainsDBConditions::prefillDBGains()
{
  // const int MAX_SIZE = 273024;//for ME1a unganged
  const int MAX_SIZE = 252288;
  const int FACTOR = 1000;
  const int MAX_SHORT = 32767;
  CSCDBGains * cndbgains = new CSCDBGains();
    
  int db_index;
  float db_gainslope;//db_intercpt, db_chisq;
  std::vector<int> db_index_id;
  std::vector<float> db_slope;
  std::vector<float> db_intercept;
  std::vector<float> db_chi2;
  int new_index;
  float new_gainslope,new_intercpt, new_chisq;
  std::vector<int> new_cham_id;
  std::vector<int> new_index_id;
  std::vector<int> new_strips;
  std::vector<float> new_slope;
  std::vector<float> new_intercept;
  std::vector<float> new_chi2;

  int counter;
  int db_nrlines=0;
  int new_nrlines=0;
  
  std::ifstream dbdata; 
  dbdata.open("old_dbgains.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: old_dbgains.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!dbdata.eof() ) { 
    dbdata >> db_index >> db_gainslope; 
    db_index_id.push_back(db_index);
    db_slope.push_back(db_gainslope);
    //db_intercept.push_back(db_intercpt);
    //db_chi2.push_back(db_chisq);
    db_nrlines++;
  }
  dbdata.close();

  std::ifstream newdata;
  newdata.open("gains.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: gains.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_index >> new_gainslope >> new_intercpt >> new_chisq ; 
    new_index_id.push_back(new_index);
    new_slope.push_back(new_gainslope);
    new_intercept.push_back(new_intercpt);
    new_chi2.push_back(new_chisq);
    new_nrlines++;
  }
  newdata.close();

  CSCDBGains::GainContainer & itemvector = cndbgains->gains;
  itemvector.resize(MAX_SIZE);
  cndbgains->factor_gain = (short int) (FACTOR);
  std::cout<<" myfactor "<<cndbgains->factor_gain<<std::endl;

  for(int i=0; i<MAX_SIZE;++i){
    itemvector[i].gain_slope= int (db_slope[i]*FACTOR+0.5);
  }

   for(int i=0; i<MAX_SIZE;++i){
     counter=db_index_id[i]; 
     itemvector[i] = itemvector[counter];
     itemvector[i].gain_slope = int (db_slope[i]);
 
     for (unsigned int k=0;k<new_index_id.size()-1;k++){
       if(counter==new_index_id[k]){
	 if ((short int) (fabs(new_slope[k]*FACTOR+0.5))<MAX_SHORT) itemvector[counter].gain_slope= int (new_slope[k]*FACTOR+0.5);
	 itemvector[i] = itemvector[counter];
       }
     }
     if(counter>223968){
       itemvector[counter].gain_slope = int (db_slope[i]);
       itemvector[i] = itemvector[counter];
     }
   }
     return cndbgains;
}


#endif

