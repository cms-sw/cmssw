#ifndef _CSCGAINSCONDITIONS_H
#define _CSCGAINSCONDITIONS_H

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
#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"

class CSCGainsConditions: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCGainsConditions(const edm::ParameterSet&);
  ~CSCGainsConditions() override;
  
  static CSCGains * prefillGains();

  typedef const  CSCGains * ReturnType;
  
  ReturnType produceGains(const CSCGainsRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & ) override;
  CSCGains *cnGains ;

};

#endif
