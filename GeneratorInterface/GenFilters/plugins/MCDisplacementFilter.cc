
// -*- C++ -*-
//
// Package:    MCDisplacementFilter
// Class:      MCDisplacementFilter
//

// system include files
//#include <memory>
//#include <iostream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

class MCDisplacementFilter : public edm::global::EDFilter<> {
public:
  explicit MCDisplacementFilter(const edm::ParameterSet&);
  ~MCDisplacementFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------
  edm::InputTag hepMCProductTag_;
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  // To chose on which pdgIds the filter is applied - if ParticleIDs.at(0)==0 runs on all particles
  std::vector<int> particleIDs_;

  const float theUpperCut_;  // Maximum displacement accepted
  const float theLowerCut_;  // Minimum displacement accepted
};

//methods implementation
//
//Class initialization
MCDisplacementFilter::MCDisplacementFilter(const edm::ParameterSet& iConfig)
    : hepMCProductTag_(iConfig.getParameter<edm::InputTag>("hepMCProductTag")),
      token_(consumes<edm::HepMCProduct>(hepMCProductTag_)),
      particleIDs_(iConfig.getParameter<std::vector<int>>("ParticleIDs")),
      theUpperCut_(iConfig.getParameter<double>("LengMax")),
      theLowerCut_(iConfig.getParameter<double>("LengMin")) {}

//Filter description
void MCDisplacementFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hepMCProductTag", edm::InputTag("generator", "unsmeared"));
  desc.add<std::vector<int>>("ParticleIDs", std::vector<int>{0});
  desc.add<double>("LengMax", -1.);
  desc.add<double>("LengMin", -1.);
  descriptions.addDefault(desc);
}

// ------------ method called to skim the data  ------------
bool MCDisplacementFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  Handle<HepMCProduct> evt;

  iEvent.getByToken(token_, evt);

  bool pass = false;
  bool matchedID = true;

  const float theUpperCut2 = theUpperCut_ * theUpperCut_;
  const float theLowerCut2 = theLowerCut_ * theLowerCut_;

  const HepMC::GenEvent* generated_event = evt->GetEvent();
  HepMC::GenEvent::particle_const_iterator p;

  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); p++) {
    //matchedID might be moved to false to true for a particle in the event, it needs to be resetted everytime
    if (particleIDs_.at(0) != 0)
      matchedID = false;

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
        // lower/upper cuts can be also <= 0 - prompt particle needs to be accepted in that case
        if ((theLowerCut_ <= 0. || dist2 >= theLowerCut2) && (theUpperCut_ <= 0. || dist2 < theUpperCut2)) {
          pass = true;
          break;
        }
      }
      if (((*p)->production_vertex() == nullptr) && (((*p)->end_vertex() != nullptr))) {
        // lower/upper cuts can be also 0 - prompt particle needs to be accepted in that case
        float distEndVert = (*p)->end_vertex()->position().perp();
        if ((theLowerCut_ <= 0. || distEndVert >= theLowerCut_) && (theUpperCut_ <= 0. || distEndVert < theUpperCut_)) {
          pass = true;
          break;
        }
      }
    }
  }

  return pass;
}

DEFINE_FWK_MODULE(MCDisplacementFilter);
