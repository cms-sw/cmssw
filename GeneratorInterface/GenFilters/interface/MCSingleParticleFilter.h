#ifndef MCSingleParticleFilter_h
#define MCSingleParticleFilter_h
// -*- C++ -*-
//
// Package:    MCSingleParticleFilter
// Class:      MCSingleParticleFilter
//
/* 

 Description: filter events based on the Pythia particleID and the Pt_hat

 Implementation: inherits from generic EDFilter
     
*/
//
// Original Author:  Filip Moortgat
//         Created:  Mon Sept 11 10:57:54 CET 2006
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//
namespace edm {
  class HepMCProduct;
}

class MCSingleParticleFilter : public edm::global::EDFilter<> {
public:
  explicit MCSingleParticleFilter(const edm::ParameterSet&);
  ~MCSingleParticleFilter() override;

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------memeber function----------------------
  // ----------member data ---------------------------

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> particleID;
  std::vector<double> ptMin;
  std::vector<double> etaMin;
  std::vector<double> etaMax;
  std::vector<int> status;
  const double betaBoost;
};
#endif
