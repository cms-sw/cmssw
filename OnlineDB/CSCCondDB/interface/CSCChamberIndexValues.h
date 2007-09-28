#ifndef _CSCCHAMBERINDEXVALUES_H
#define _CSCCHAMBERINDEXVALUES_H

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
#include "CondFormats/CSCObjects/interface/CSCChamberIndex.h"
#include "CondFormats/DataRecord/interface/CSCChamberIndexRcd.h"

class CSCChamberIndexValues: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCChamberIndexValues(const edm::ParameterSet&);
  ~CSCChamberIndexValues();
  
  void fillChamberIndex();

  typedef const  CSCChamberIndex * ReturnType;
  
  ReturnType produceChamberIndex(const CSCChamberIndexRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCChamberIndex *mapobj ;

};

#endif
