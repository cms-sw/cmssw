// -*- C++ -*-
//
// Package:    MCLongLivedParticles
// Class:      MCLongLivedParticles
// 
 

// Description: 
//Filter particles based on their minimum and/or maximum displacement on the transverse plane and optionally on their pdgIds
//To run independently of pdgIds, run with particleIDs_(0) entry in filter declaration

// Implementation: inherits from generic EDFilter
     
// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class decleration
//

namespace edm {
  class HepMCProduct;
}

class MCLongLivedParticles : public edm::EDFilter {
public:
  explicit MCLongLivedParticles(const edm::ParameterSet&);
  ~MCLongLivedParticles() override = default;
 
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions); 

  bool filter(edm::Event&, const edm::EventSetup&) override;
private:
  // ----------member data ---------------------------
  edm::InputTag hepMCProductTag_;
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  //possible now to chose on which pdgIds the filter is applied - if ParticleIDs.size()==0 runs on all particles in  the event as the preovious filter version
  std::vector<int> particleIDs_; 
  float theCut; 
  float theUpperCut_; // Maximum displacement accepted
  float theLowerCut_; //Minimum displacement accepted 
};
