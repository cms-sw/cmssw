#ifndef _CSCFRONTIERCROSSTALKCONDITIONS_H
#define _CSCFRONTIERCROSSTALKCONDITIONS_H

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

class CSCFakeCrosstalkConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCFakeCrosstalkConditions(const edm::ParameterSet&);
  ~CSCFakeCrosstalkConditions();
  
  float mean,min,minchi;
  int seed;long int M;

  void prefillCrosstalk();
  
  typedef const  CSCcrosstalk * ReturnType;
  
  ReturnType produceCrosstalk(const CSCcrosstalkRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCcrosstalk *cncrosstalk ;
  
};

#endif

