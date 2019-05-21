#ifndef MCDijetResonance_h
#define MCDijetResonance_h
// -*- C++ -*-
//
// Package:    MCDijetResonance
// Class:      MCDijetResonance
//
/* 

 Description: filter to select Dijet Resonance events.

 Implementation: inherits from generic EDFilter
     
*/
//
// Author: Robert Harris
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//
namespace edm {
  class HepMCProduct;
}

class MCDijetResonance : public edm::EDFilter {
public:
  explicit MCDijetResonance(const edm::ParameterSet&);
  ~MCDijetResonance() override;
  void endJob() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::string dijetProcess;
  unsigned int nEvents;
  unsigned int nAccepted;
  int maxQuarkID;
  int bosonID;
};
#endif
