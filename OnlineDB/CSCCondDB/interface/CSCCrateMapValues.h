#ifndef _CSCCRATEMAPVALUES_H
#define _CSCCRATEMAPVALUES_H

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
#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"

class CSCCrateMapValues: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCCrateMapValues(const edm::ParameterSet&);
  ~CSCCrateMapValues();
  
  void fillCrateMap();

  typedef const  CSCCrateMap * ReturnType;
  
  ReturnType produceCrateMap(const CSCCrateMapRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCCrateMap *mapobj ;

};

#endif
