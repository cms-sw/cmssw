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
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCCrosstalkConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCCrosstalkConditions(const edm::ParameterSet&);
  ~CSCCrosstalkConditions() override;
  

  static  CSCcrosstalk * prefillCrosstalk();
  
  typedef const  CSCcrosstalk * ReturnType;
  
  ReturnType produceCrosstalk(const CSCcrosstalkRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & ) override;
  CSCcrosstalk *cnCrosstalk ;
  
};

#endif
