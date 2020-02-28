// -*- C++ -*-
//
// Package:    L1ScalesProducer
// Class:      L1MuScalesTester
//

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class L1MuScale;
//
// class decleration
//

class L1MuScalesTester : public edm::EDAnalyzer {
public:
  explicit L1MuScalesTester(const edm::ParameterSet&);
  ~L1MuScalesTester() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void printScale(const L1MuScale*);

private:
  // ----------member data ---------------------------
};
