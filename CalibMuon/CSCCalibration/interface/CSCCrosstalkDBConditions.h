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

  void prefillDBCrosstalk();

  typedef const  CSCDBCrosstalk * ReturnType;
  
  ReturnType produceDBCrosstalk(const CSCDBCrosstalkRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCDBCrosstalk *cndbcrosstalk ;

};

#endif
