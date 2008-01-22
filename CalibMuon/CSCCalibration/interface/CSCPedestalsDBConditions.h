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
  
  void prefillDBPedestals();

  typedef const  CSCDBPedestals * ReturnType;
  
  ReturnType produceDBPedestals(const CSCDBPedestalsRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCDBPedestals*cndbpedestals ;

};

#endif
