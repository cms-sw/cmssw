#ifndef _CSCPEDESTALSDBCONDITIONS_H
#define _CSCPEDESTALSDBCONDITIONS_H

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
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"

class CSCPedestalsDBConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCPedestalsDBConditions(const edm::ParameterSet&);
  ~CSCPedestalsDBConditions();
  
  inline static CSCDBPedestals * prefillDBPedestals();

  typedef const  CSCDBPedestals * ReturnType;
  
  ReturnType produceDBPedestals(const CSCDBPedestalsRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCDBPedestals *cndbPedestals ;

};

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCDBPedestals * CSCPedestalsDBConditions::prefillDBPedestals()  
{
 CSCDBPedestals * cndbpedestals = new CSCDBPedestals();

  float mean,min,minchi;
  int seed;long int M;
  int new_chamber_id,db_index;
  float db_ped, db_rms;
  std::vector<int> db_index_id;
  std::vector<float> db_peds;
  std::vector<float> db_pedrms;
  int new_index;
  float new_ped,new_rms;
  std::vector<int> new_index_id;
  std::vector<float> new_peds;
  std::vector<float> new_pedrms;

  int counter;
  int db_nrlines=0;
  int new_nrlines=0;
  seed = 10000;	
  srand(seed);
  mean=6.8, min=-10.0, minchi=1.0, M=1000;
  
  std::ifstream dbdata; 
  dbdata.open("dbpeds.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: dbpeds.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!dbdata.eof() ) { 
    dbdata >> db_index >> db_ped >> db_rms ; 
    db_index_id.push_back(db_index);
    db_peds.push_back(db_ped);
    db_pedrms.push_back(db_rms);
    db_nrlines++;
  }
  dbdata.close();

  std::ifstream newdata;
  newdata.open("new_peds.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: new_peds.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_index >> new_ped >> new_rms ; 
    new_index_id.push_back(new_index);
    new_peds.push_back(new_ped);
    new_pedrms.push_back(new_rms);
    new_nrlines++;
  }
  newdata.close();
  
  std::vector<CSCDBPedestals::Item> itemvector;
  itemvector.resize(252288);
  // itemvector.resize(217728);

  //for(int i=0; i<217728;++i){
  for(int i=0; i<252288;++i){
    itemvector[i].ped= db_peds[i];
    itemvector[i].rms= db_pedrms[i];
  }

  //for(int i=0; i<217728;++i){
  for(int i=0; i<252288;++i){
     counter=db_index_id[i];  
     for (unsigned int k=0;k<new_index_id.size()-1;k++){
       if(counter==new_index_id[k]){
	 itemvector[counter].ped= new_peds[k];
	 itemvector[counter].rms= new_pedrms[k];
	 itemvector[i] = itemvector[counter];
	//std::cout<<"counter "<<counter<<" new_index_id[k] "<<new_index_id[k]<<" new_slope[k] "<<new_slope[k]<<" db_slope[k] "<<db_slope[k]<<std::endl;
       }  
     }
   }
   return cndbpedestals;
  //std::copy(itemvector.begin(), itemvector.end(), std::back_inserter(cndbpedestals->pedestals));
}

#endif
