
// -*- C++ -*-
//
// Package:    MCLongLivedParticles
// Class:      MCLongLivedParticles
//

// system include files
#include <memory>
#include <iostream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
//#include "GeneratorInterface/GenFilters/plugins/MCLongLivedParticles.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

using namespace edm;
using namespace std;

//Filter particles based on their minimum and/or maximum displacement on the transverse plane and optionally on their pdgIds
//To run independently of pdgId, do not insert the particleIDs entry in filter declaration


// class decleration
//
namespace edm {
  class HepMCProduct;
  class ConfigurationDescriptions;
}  // namespace edm

class MCLongLivedParticles : public edm::global::EDFilter<> {
public:
  explicit MCLongLivedParticles(const edm::ParameterSet&);
  ~MCLongLivedParticles() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------
  edm::InputTag hepMCProductTag_;
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> particleIDs_;  // To chose on which pdgIds the filter is applied - if ParticleIDs.at(0)==0 runs on all particles

  const float theUpperCut_;  // Maximum displacement accepted
  const float theLowerCut_;  // Minimum displacement accepted
};


//methods implementation
//
//Class initialization
MCLongLivedParticles::MCLongLivedParticles(const edm::ParameterSet& iConfig)
    : hepMCProductTag_(iConfig.getParameter<edm::InputTag>("hepMCProductTag")),
      token_(consumes<edm::HepMCProduct>(hepMCProductTag_)),
      particleIDs_(iConfig.getParameter<std::vector<int>>("ParticleIDs")),
      theUpperCut_(iConfig.getParameter<double>("LengMax")),
      theLowerCut_(iConfig.getParameter<double>("LengMin")) {}

//Filter description
void MCLongLivedParticles::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hepMCProductTag", edm::InputTag("generator","unsmeared"));
  desc.add<std::vector<int>>("ParticleIDs", std::vector<int>{0});
  desc.add<double>("LengMax", -1.);
  desc.add<double>("LengMin", -1.);
  descriptions.add("selectsDisplacement", desc);
}

// ------------ method called to skim the data  ------------
bool MCLongLivedParticles::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  Handle<HepMCProduct> evt;

  iEvent.getByToken(token_, evt);

  bool pass = false;
  bool matchedID = true;

  float theUpperCut2 = theUpperCut_ * theUpperCut_;
  float theLowerCut2 = theLowerCut_ * theLowerCut_;

  if (particleIDs_.at(0) != 0)
    matchedID = false;

  const HepMC::GenEvent* generated_event = evt->GetEvent();
  HepMC::GenEvent::particle_const_iterator p;

  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); p++) {
    //if a list of pdgId is provided, loop only on particles with those pdgId

    for (unsigned int idx = 0; idx < particleIDs_.size(); idx++) {
      if (abs((*p)->pdg_id()) == abs(particleIDs_.at(idx))) {  //compares absolute values of pdgIds
        matchedID = true;
        break;
      }
    }

    if (matchedID) {
      if (theLowerCut_ <= 0. && theUpperCut_ <= 0.) {
        pass = true;
        break;
      }
      if (((*p)->production_vertex() != nullptr) && ((*p)->end_vertex() != nullptr)) {
        float dist2 = (((*p)->production_vertex())->position().x() - ((*p)->end_vertex())->position().x()) *
                          (((*p)->production_vertex())->position().x() - ((*p)->end_vertex())->position().x()) +
                      (((*p)->production_vertex())->position().y() - ((*p)->end_vertex())->position().y()) *
                          (((*p)->production_vertex())->position().y() - ((*p)->end_vertex())->position().y());

        if ((dist2 >= theLowerCut2 || theLowerCut_ <= 0.) &&
            (dist2 < theUpperCut2 ||
             theUpperCut_ <= 0.)) {  //lower cut can be also 0 - prompt particle needs to be accepted in that case
          pass = true;
          break;
        }
      }
      if (((*p)->production_vertex() == nullptr) && (!((*p)->end_vertex() != nullptr))) {
        if ((((*p)->end_vertex()->position().perp() >= theLowerCut_) || theLowerCut_ <= 0.) &&
            (((*p)->end_vertex()->position().perp() < theUpperCut_) ||
             theUpperCut_ <= 0.)) {  // lower cut can be also 0 - prompt particle needs to be accepted in that case
          pass = true;
          break;
        }
      }
    }
  }

  return pass;
}
DEFINE_FWK_MODULE(MCLongLivedParticles);
