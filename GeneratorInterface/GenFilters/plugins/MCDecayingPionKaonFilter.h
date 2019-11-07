#ifndef MCDecayingPionKaonFilter_h
#define MCDecayingPionKaonFilter_h
// -*- C++ -*-
//
// Package:    MCDecayingPionKaonFilter
// Class:      MCDecayingPionKaonFilter
//
/* 

 Description: filter events based on the Pythia particleID and the Pt_hat

 Implementation: inherits from generic EDFilter
     
*/
//
// Original Author:  Juan Alcaraz (13/03/2008)
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

class MCDecayingPionKaonFilter : public edm::EDFilter {
public:
  explicit MCDecayingPionKaonFilter(const edm::ParameterSet&);
  ~MCDecayingPionKaonFilter() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> particleID;
  std::vector<double> ptMin;
  std::vector<double> etaMin;
  std::vector<double> etaMax;
  std::vector<double> decayRadiusMin;
  std::vector<double> decayRadiusMax;
  std::vector<double> decayZMin;
  std::vector<double> decayZMax;
  double ptMuMin;
};
#endif
