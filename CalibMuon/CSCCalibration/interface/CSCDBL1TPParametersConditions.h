#ifndef _CSCDBL1TPPARAMETERSCONDITIONS_H
#define _CSCDBL1TPPARAMETERSCONDITIONS_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <cmath>
#include <memory>

#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"
#include "CondFormats/DataRecord/interface/CSCDBL1TPParametersRcd.h"
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class CSCDBL1TPParametersConditions : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  CSCDBL1TPParametersConditions(const edm::ParameterSet &);
  ~CSCDBL1TPParametersConditions();

  inline static CSCDBL1TPParameters *prefillCSCDBL1TPParameters();

  typedef std::unique_ptr<CSCDBL1TPParameters> ReturnType;

  ReturnType produceCSCDBL1TPParameters(const CSCDBL1TPParametersRcd &);

private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue &, edm::ValidityInterval &);
};

#include <fstream>
#include <iostream>
#include <vector>

// to workaround plugin library
inline CSCDBL1TPParameters *CSCDBL1TPParametersConditions::prefillCSCDBL1TPParameters() {
  CSCDBL1TPParameters *cnl1tp = new CSCDBL1TPParameters();

  cnl1tp->setAlctFifoTbins(16);
  cnl1tp->setAlctFifoPretrig(10);
  cnl1tp->setAlctDriftDelay(2);
  cnl1tp->setAlctNplanesHitPretrig(3);  // was 2, new is 3
  cnl1tp->setAlctNplanesHitPattern(4);
  cnl1tp->setAlctNplanesHitAccelPretrig(3);  // was 2, new is 3
  cnl1tp->setAlctNplanesHitAccelPattern(4);
  cnl1tp->setAlctTrigMode(2);
  cnl1tp->setAlctAccelMode(0);
  cnl1tp->setAlctL1aWindowWidth(7);

  cnl1tp->setClctFifoTbins(12);
  cnl1tp->setClctFifoPretrig(7);
  cnl1tp->setClctHitPersist(4);  // was 6, new is 4
  cnl1tp->setClctDriftDelay(2);
  cnl1tp->setClctNplanesHitPretrig(3);  // was 2, new is 3
  cnl1tp->setClctNplanesHitPattern(4);
  cnl1tp->setClctPidThreshPretrig(2);
  cnl1tp->setClctMinSeparation(10);

  // the new parameters
  cnl1tp->setTmbMpcBlockMe1a(0);
  cnl1tp->setTmbAlctTrigEnable(0);
  cnl1tp->setTmbClctTrigEnable(0);
  cnl1tp->setTmbMatchTrigEnable(1);
  cnl1tp->setTmbMatchTrigWindowSize(7);
  cnl1tp->setTmbTmbL1aWindowSize(7);

  return cnl1tp;
}

#endif
