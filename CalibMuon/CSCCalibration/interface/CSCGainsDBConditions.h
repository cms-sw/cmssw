#ifndef _CSCGAINSDBCONDITIONS_H
#define _CSCGAINSDBCONDITIONS_H

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
  CSCDBGains * cndbgains = new CSCDBGains();

  float mean,min,minchi;
  int seed;long int M;
  int new_chamber_id,db_index,new_strip;
  float db_gainslope,db_intercpt, db_chisq;
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
  seed = 10000;	
  srand(seed);
  mean=6.8, min=-10.0, minchi=1.0, M=1000;
  
  std::ifstream dbdata; 
  dbdata.open("dbgains.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: dbgains.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!dbdata.eof() ) { 
    dbdata >> db_index >> db_gainslope >> db_intercpt >> db_chisq ; 
    db_index_id.push_back(db_index);
    db_slope.push_back(db_gainslope);
    db_intercept.push_back(db_intercpt);
    db_chi2.push_back(db_chisq);
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
    //new_cham_id.push_back(new_chamber_id);
    new_index_id.push_back(new_index);
    //new_strips.push_back(new_strip);
    new_slope.push_back(new_gainslope);
    new_intercept.push_back(new_intercpt);
    new_chi2.push_back(new_chisq);
    new_nrlines++;
  }
  newdata.close();
  
  std::vector<CSCDBGains::Item> itemvector;
   itemvector.resize(252288);

   for(int i=0; i<252288;++i){
     // for(int i=0; i<217728;++i){
    itemvector[i].gain_slope= db_slope[i];
    itemvector[i].gain_intercept= db_intercept[i];
    itemvector[i].gain_chi2= db_chi2[i];
  }

   for(int i=0; i<252288;++i){
     //for(int i=0; i<217728;++i){
     counter=db_index_id[i];  
     for (unsigned int k=0;k<new_index_id.size()-1;k++){
       if(counter==new_index_id[k]){
	 itemvector[counter].gain_slope= new_slope[k];
	 itemvector[counter].gain_intercept= new_intercept[k];
	 itemvector[counter].gain_chi2= new_chi2[k];
	 itemvector[i] = itemvector[counter];
	//std::cout<<"counter "<<counter<<" new_index_id[k] "<<new_index_id[k]<<" new_slope[k] "<<new_slope[k]<<" db_slope[k] "<<db_slope[k]<<std::endl;
       }  
     }
   }
   
   return cndbgains;
   //   std::copy(itemvector.begin(), itemvector.end(), std::back_inserter(cndbgains->gains));
}


#endif
