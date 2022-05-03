// -*- C++ -*-
//
// Package:    MCLongLivedParticles
// Class:      MCLongLivedParticles
//
/* 

 Description: 
Filter particles based on their minimum and/or maximum displacement on the transverse plane and optionally on their pdgIds
To run independently of pdgIds, do not insert the particleIDs entry in filter declaration

*/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

namespace edm {
  class HepMCProduct;
  class ConfigurationDescriptions;
}  // namespace edm

class MCLongLivedParticles : public edm::global::EDFilter<> {
public:
  explicit MCLongLivedParticles(const edm::ParameterSet&);
  // ~MCLongLivedParticles() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------
  std::string moduleLabel_;
  edm::InputTag hepMCProductTag_;
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int>
      particleIDs;  //possible now to chose on which pdgIds the filter is applied - if ParticleIDs.size()==0 runs on all particles in  the event as the preovious filter version

  const float theUpperCut_;  // Maximum displacement accepted
  const float theLowerCut_;  // Minimum displacement accepted
};
