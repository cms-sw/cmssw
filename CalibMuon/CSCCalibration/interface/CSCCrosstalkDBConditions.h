#ifndef _CSCCROSSTALKDBCONDITIONS_H
#define _CSCCROSSTALKDBCONDITIONS_H

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
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"

class CSCCrosstalkDBConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCCrosstalkDBConditions(const edm::ParameterSet&);
  ~CSCCrosstalkDBConditions();
  

  inline static CSCDBCrosstalk * prefillDBCrosstalk();

  typedef const  CSCDBCrosstalk * ReturnType;
  
  ReturnType produceDBCrosstalk(const CSCDBCrosstalkRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );


  CSCDBCrosstalk *cndbCrosstalk ;

};



#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCDBCrosstalk *  CSCCrosstalkDBConditions::prefillDBCrosstalk()
{
  CSCDBCrosstalk * cndbcrosstalk = new CSCDBCrosstalk();
  //const CSCDetId& detId = CSCDetId();


  float mean,min,minchi;
  int seed;long int M;
  int db_index,new_chamber_id,new_strip,new_index;
  float db_slope_right,db_slope_left,db_intercept_right;
  float db_intercept_left, db_chi2_right,db_chi2_left;
  std::vector<int> db_index_id;
  std::vector<float> db_slope_r;
  std::vector<float> db_intercept_r;
  std::vector<float> db_chi2_r;
  std::vector<float> db_slope_l;
  std::vector<float> db_intercept_l;
  std::vector<float> db_chi2_l;
  float new_slope_right,new_slope_left,new_intercept_right;
  float new_intercept_left, new_chi2_right,new_chi2_left;
  std::vector<int> new_cham_id;
  std::vector<int> new_index_id;
  std::vector<int> new_strips;
  std::vector<float> new_slope_r;
  std::vector<float> new_intercept_r;
  std::vector<float> new_chi2_r;
  std::vector<float> new_slope_l;
  std::vector<float> new_intercept_l;
  std::vector<float> new_chi2_l;


      
  int counter;
  int db_nrlines=0;
  int new_nrlines=0;
  seed = 10000;	
  srand(seed);
  mean=-0.0009, min=0.035, minchi=1.5, M=1000;
    
  std::ifstream dbdata; 
  dbdata.open("dbxtalk.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: dbxtalk.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!dbdata.eof() ) { 
    dbdata >> db_index >> db_slope_right >> db_intercept_right >> db_chi2_right >> db_slope_left >> db_intercept_left >> db_chi2_left ;
    db_index_id.push_back(db_index);
    db_slope_r.push_back(db_slope_right);
    db_slope_l.push_back(db_slope_left);
    db_intercept_r.push_back(db_intercept_right);
    db_intercept_l.push_back(db_intercept_left);
    db_chi2_r.push_back(db_chi2_right);
    db_chi2_l.push_back(db_chi2_left); 
    db_nrlines++;
  }
  dbdata.close();

  std::ifstream newdata;
  newdata.open("xtalk.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: xtalk.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_index >> new_slope_right >> new_intercept_right >> new_chi2_right >> new_slope_left >> new_intercept_left >> new_chi2_left ; 
    // new_cham_id.push_back(new_chamber_id);
    new_index_id.push_back(new_index);
    //new_strips.push_back(new_strip);
    new_slope_r.push_back(new_slope_right);
    new_slope_l.push_back(new_slope_left);
    new_intercept_r.push_back(new_intercept_right);
    new_intercept_l.push_back(new_intercept_left);
    new_chi2_r.push_back(new_chi2_right);
    new_chi2_l.push_back(new_chi2_left);
    new_nrlines++;
  }
  newdata.close();
  
  CSCDBCrosstalk::CrosstalkContainer & itemvector = cndbcrosstalk->crosstalk;
  itemvector.resize(252288);
  //itemvector.resize(217728);
  for(int i=0; i<252288;++i){
    //for(int i=0; i<217728;++i){
    itemvector[i].xtalk_slope_right=db_slope_r[i];
    itemvector[i].xtalk_intercept_right=db_intercept_r[i]; 
    itemvector[i].xtalk_chi2_right=db_chi2_r[i];
    itemvector[i].xtalk_slope_left=db_slope_l[i];  
    itemvector[i].xtalk_intercept_left=db_intercept_l[i];  
    itemvector[i].xtalk_chi2_left=db_chi2_l[i];
  }

  for(int i=0; i<252288;++i){
    //for(int i=0; i<217728;++i){
    counter=db_index_id[i];  
    for (unsigned int k=0;k<new_index_id.size()-1;k++){
      if(counter==new_index_id[k]){
	itemvector[counter].xtalk_slope_right=new_slope_r[k];
	itemvector[counter].xtalk_intercept_right=new_intercept_r[k]; 
	itemvector[counter].xtalk_chi2_right=new_chi2_r[k];
	itemvector[counter].xtalk_slope_left=new_slope_l[k];  
	itemvector[counter].xtalk_intercept_left=new_intercept_l[k];  
	itemvector[counter].xtalk_chi2_left=new_chi2_l[k];
	itemvector[i] = itemvector[counter];
	//std::cout<<" counter "<<counter <<" dbindex "<<new_index_id[k]<<" dbslope " <<db_slope_r[k]<<" new slope "<<new_slope_r[k]<<std::endl;
      }  
    }
  }
  

  return cndbcrosstalk;

}
  




#endif
