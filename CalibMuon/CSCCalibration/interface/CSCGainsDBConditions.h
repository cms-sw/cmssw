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
  
  void prefillDBGains();

  typedef const  CSCDBGains * ReturnType;
  
  ReturnType produceDBGains(const CSCDBGainsRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCDBGains *cndbgains ;

};

#endif
