// -*- C++ -*-
//
// Package:    L1ScalesProducer
// Class:      L1MuScalesTester
//

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/DataRecord/interface/L1MuGMTScalesRcd.h"

class L1MuScale;
//
// class decleration
//

class L1MuScalesTester : public edm::one::EDAnalyzer<> {
public:
  explicit L1MuScalesTester(const edm::ParameterSet&);

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void printScale(const L1MuScale*);

private:
  // ----------member data ---------------------------
  edm::ESGetToken<L1MuTriggerScales, L1MuTriggerScalesRcd> l1muscalesToken_;
  edm::ESGetToken<L1MuTriggerPtScale, L1MuTriggerPtScaleRcd> l1muptscaleToken_;
  edm::ESGetToken<L1MuGMTScales, L1MuGMTScalesRcd> l1gmtscalesToken_;
};
