// -*- C++ -*-
//
// Package:    FourLepFilter
// Class:      FourLepFilter
//
/**\class FourLepFilter FourLepFilter.cc psi2s1s/FourLepFilter/src/FourLepFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  bian jianguo
//         Created:  Tue Nov 22 20:39:54 CST 2011
//
//

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>
#include <cstdlib>
#include <string>

//
// class declaration
//

class FourLepFilter : public edm::global::EDFilter<> {
public:
  explicit FourLepFilter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  const edm::EDGetToken token_;
  const double minPt;
  const double maxEta;
  const double maxPt;
  const double minEta;
  const int particleID;
};

FourLepFilter::FourLepFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      minPt(iConfig.getUntrackedParameter("MinPt", 0.)),
      maxEta(iConfig.getUntrackedParameter("MaxEta", 10.)),
      maxPt(iConfig.getUntrackedParameter("MaxPt", 1000.)),
      minEta(iConfig.getUntrackedParameter("MinEta", 0.)),
      particleID(iConfig.getUntrackedParameter("ParticleID", 0)) {}

// ------------ method called on each new Event  ------------
bool FourLepFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  int nLeptons = 0;

  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if ((*p)->status() != 1)
      continue;
    if ((*p)->momentum().perp() > minPt && std::fabs((*p)->momentum().eta()) < maxEta &&
        (*p)->momentum().perp() < maxPt && std::fabs((*p)->momentum().eta()) > minEta) {
      if (std::abs((*p)->pdg_id()) == particleID)
        nLeptons++;
    }
    if (nLeptons >= 4) {
      accepted = true;
      break;
    }
  }
  return accepted;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void FourLepFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(FourLepFilter);
