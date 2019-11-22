#ifndef MCZll_h
#define MCZll_h
// -*- C++ -*-
//
// Package:    MCZll
// Class:      MCZll
//
/* 

 Description: filter events based on the Pythia ProcessID and the Pt_hat

 Implementation: inherits from generic EDFilter
     
*/
//
// Original Author:  Paolo Meridiani
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

class MCZll : public edm::EDFilter {
public:
  explicit MCZll(const edm::ParameterSet&);
  ~MCZll() override;
  void endJob() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  edm::EDGetTokenT<edm::HepMCProduct> token_;
  int leptonFlavour_;
  double leptonPtMin_;
  double leptonPtMax_;
  double leptonEtaMin_;
  double leptonEtaMax_;
  std::pair<double, double> zMassRange_;
  unsigned int nEvents_;
  unsigned int nAccepted_;
  bool filter_;
};
#endif
