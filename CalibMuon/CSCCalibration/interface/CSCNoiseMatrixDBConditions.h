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
 
  void prefillDBNoiseMatrix();

  typedef const  CSCDBNoiseMatrix * ReturnType;
  
  ReturnType produceDBNoiseMatrix(const CSCDBNoiseMatrixRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCDBNoiseMatrix *cndbmatrix ;

};

#endif
