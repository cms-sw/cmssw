#ifndef MCLongLivedParticles_h
#define MCLongLivedParticles_h
// -*- C++ -*-
//
// Package:    MCLongLivedParticles
// Class:      MCLongLivedParticles
//
/* 

 Description: 
Filter particles based on their minimum and/or maximum displacement on the transverse plane and optionally on their pdgIds
To run independently of pdgIds, do not insert the particleIDs entry in filter declaration

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
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Pythia8/Pythia.h"

//
// class decleration
//

namespace edm {
  class HepMCProduct;
}

class MCLongLivedParticles : public edm::global::EDFilter<> {
public:
  explicit MCLongLivedParticles(const edm::ParameterSet&);
  ~MCLongLivedParticles() override = default;

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int>
      particleIDs;  //possible now to chose on which pdgIds the filter is applied - if ParticleIDs.size()==0 runs on all particles in  the event as the preovious filter version

  float theUpperCut;  // Maximum displacement accepted
  float theLowerCut;  //Minimum displacement accepted
  edm::InputTag hepMCProductTag_;
};
#endif
