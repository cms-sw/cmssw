#ifndef _CSCCROSSTALKCONDITIONS_H
#define _CSCCROSSTALKCONDITIONS_H

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

#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCCrosstalkConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCCrosstalkConditions(const edm::ParameterSet&);
  ~CSCCrosstalkConditions();
  
  float mean,min,minchi;
  int seed;long int M;
  int old_chamber_id,old_strip,new_chamber_id,new_strip;
  float old_slope_right,old_slope_left,old_intercept_right;
  float old_intercept_left, old_chi2_right,old_chi2_left;
  std::vector<int> old_cham_id;
  std::vector<int> old_strips;
  std::vector<float> old_slope_r;
  std::vector<float> old_intercept_r;
  std::vector<float> old_chi2_r;
  std::vector<float> old_slope_l;
  std::vector<float> old_intercept_l;
  std::vector<float> old_chi2_l;
  float new_slope_right,new_slope_left,new_intercept_right;
  float new_intercept_left, new_chi2_right,new_chi2_left;
  std::vector<int> new_cham_id;
  std::vector<int> new_strips;
  std::vector<float> new_slope_r;
  std::vector<float> new_intercept_r;
  std::vector<float> new_chi2_r;
  std::vector<float> new_slope_l;
  std::vector<float> new_intercept_l;
  std::vector<float> new_chi2_l;

  void prefillCrosstalk();
  
  typedef const  CSCcrosstalk * ReturnType;
  
  ReturnType produceCrosstalk(const CSCcrosstalkRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCcrosstalk *cncrosstalk ;
  
};

#endif
