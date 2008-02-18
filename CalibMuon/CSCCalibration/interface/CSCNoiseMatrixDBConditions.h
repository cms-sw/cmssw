#ifndef _CSCNOISEMATRIXDBCONDITIONS_H
#define _CSCNOISEMATRIXDBCONDITIONS_H

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
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"

class CSCNoiseMatrixDBConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCNoiseMatrixDBConditions(const edm::ParameterSet&);
  ~CSCNoiseMatrixDBConditions();
  
  inline static CSCDBNoiseMatrix * prefillDBNoiseMatrix();

  typedef const  CSCDBNoiseMatrix * ReturnType;
  
  ReturnType produceDBNoiseMatrix(const CSCDBNoiseMatrixRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCDBNoiseMatrix *cndbMatrix ;

};


#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCDBNoiseMatrix *  CSCNoiseMatrixDBConditions::prefillDBNoiseMatrix()
{
  int db_chamber_id,db_strip,new_chamber_id,new_strip,new_index, db_index;
  float db_elm33,db_elm34, db_elm44, db_elm35, db_elm45, db_elm55;
  float db_elm46, db_elm56, db_elm66, db_elm57, db_elm67, db_elm77;
  std::vector<int> db_index_id;
  std::vector<float> db_elem33;
  std::vector<float> db_elem34;
  std::vector<float> db_elem44;
  std::vector<float> db_elem45;
  std::vector<float> db_elem35;
  std::vector<float> db_elem55;
  std::vector<float> db_elem46;
  std::vector<float> db_elem56;
  std::vector<float> db_elem66;
  std::vector<float> db_elem57;
  std::vector<float> db_elem67;
  std::vector<float> db_elem77;


  float new_elm33,new_elm34, new_elm44, new_elm35, new_elm45, new_elm55;
  float  new_elm46, new_elm56, new_elm66, new_elm57, new_elm67, new_elm77;
  std::vector<int> new_cham_id;
  std::vector<int> new_index_id;
  std::vector<float> new_elem33;
  std::vector<float> new_elem34;
  std::vector<float> new_elem44;
  std::vector<float> new_elem45;
  std::vector<float> new_elem35;
  std::vector<float> new_elem55;
  std::vector<float> new_elem46;
  std::vector<float> new_elem56;
  std::vector<float> new_elem66;
  std::vector<float> new_elem57;
  std::vector<float> new_elem67;
  std::vector<float> new_elem77;

  CSCDBNoiseMatrix * cndbmatrix = new CSCDBNoiseMatrix();

 int counter;
  int db_nrlines=0;
  int new_nrlines=0;
    
  std::ifstream dbdata; 
  dbdata.open("dbmatrix.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: dbmatrix.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!dbdata.eof() ) { 
    dbdata >> db_index >> db_elm33 >> db_elm34 >> db_elm44 >> db_elm35 >> db_elm45 >> db_elm55 >> db_elm46 >> db_elm56 >> db_elm66 >> db_elm57 >> db_elm67 >> db_elm77 ; 
    db_index_id.push_back(db_index);
    db_elem33.push_back(db_elm33);
    db_elem34.push_back(db_elm34);
    db_elem44.push_back(db_elm44);
    db_elem35.push_back(db_elm35);
    db_elem45.push_back(db_elm45);
    db_elem55.push_back(db_elm55);
    db_elem46.push_back(db_elm46);
    db_elem56.push_back(db_elm56);
    db_elem66.push_back(db_elm66);
    db_elem57.push_back(db_elm57);
    db_elem67.push_back(db_elm67);
    db_elem77.push_back(db_elm77);
    db_nrlines++;
  }
  dbdata.close();

  std::ifstream newdata;
  newdata.open("matrix.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: matrix.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_index>> new_elm33 >> new_elm34 >> new_elm44 >> new_elm35 >> new_elm45 >> new_elm55 >> new_elm46 >> new_elm56 >> new_elm66 >> new_elm57 >> new_elm67 >> new_elm77 ; 
    //new_cham_id.push_back(new_chamber_id);
    new_index_id.push_back(new_index);
    new_elem33.push_back(new_elm33);
    new_elem34.push_back(new_elm34);
    new_elem44.push_back(new_elm44);
    new_elem35.push_back(new_elm35);
    new_elem45.push_back(new_elm45);
    new_elem55.push_back(new_elm55);
    new_elem46.push_back(new_elm46);
    new_elem56.push_back(new_elm56);
    new_elem66.push_back(new_elm66);
    new_elem57.push_back(new_elm57);
    new_elem67.push_back(new_elm67);
    new_elem77.push_back(new_elm77); 
    new_nrlines++;
  }
  newdata.close();
  
  std::vector<CSCDBNoiseMatrix::Item> itemvector;
  itemvector.resize(252288);
  //itemvector.resize(217728);
  
  for(int i=0; i<252288;++i){
  //for(int i=0; i<217728;++i){
    itemvector[i].elem33 = db_elem33[i];
    itemvector[i].elem34 = db_elem34[i]; 
    itemvector[i].elem44 = db_elem44[i];
    itemvector[i].elem35 = db_elem35[i];
    itemvector[i].elem45 = db_elem45[i];
    itemvector[i].elem55 = db_elem55[i];
    itemvector[i].elem46 = db_elem46[i];
    itemvector[i].elem56 = db_elem56[i];
    itemvector[i].elem66 = db_elem66[i];
    itemvector[i].elem57 = db_elem57[i];
    itemvector[i].elem67 = db_elem67[i];
    itemvector[i].elem77 = db_elem77[i];
  }

  
  for(int i=0; i<252288;++i){
    //for(int i=0; i<217728;++i){
    counter=db_index_id[i];  
    for (unsigned int k=0;k<new_index_id.size()-1;k++){
      if(counter==new_index_id[k]){
	itemvector[counter].elem33 = new_elem33[k];
	itemvector[counter].elem34 = new_elem34[k]; 
	itemvector[counter].elem44 = new_elem44[k];
	itemvector[counter].elem35 = new_elem35[k];
	itemvector[counter].elem45 = new_elem45[k];
	itemvector[counter].elem55 = new_elem55[k];
	itemvector[counter].elem46 = new_elem46[k];
	itemvector[counter].elem56 = new_elem56[k];
	itemvector[counter].elem66 = new_elem66[k];
	itemvector[counter].elem57 = new_elem57[k];
	itemvector[counter].elem67 = new_elem67[k];
	itemvector[counter].elem77 = new_elem77[k];
	itemvector[i] = itemvector[counter];
      }  
    }
  }
  
  return cndbmatrix;
  //std::copy(itemvector.begin(), itemvector.end(), std::back_inserter(cndbmatrix->matrix));
}

#endif
