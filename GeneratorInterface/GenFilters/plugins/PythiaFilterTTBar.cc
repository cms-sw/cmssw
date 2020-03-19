// -*- C++ -*-
//
// Package:    PythiaFilterTTBar
// Class:      PythiaFilterTTBar
//
/**\class PythiaFilterTTBar PythiaFilterTTBar.cc GeneratorInterface/GenFilter/src/PythiaFilterTTBar.cc

 Description: edmFilter to select a TTBar decay channel

 Implementation:
    decayType: 1 + leptonFlavour: 0 -> Semi-leptonic
                   leptonFlavour: 1 -> Semi-e
		   leptonFlavour: 2 -> Semi-mu
		   leptonFlavour: 3 -> Semi-tau
    decayType: 2 -> di-leptonic (no seperate channels implemented yet)

    decayType: 3 -> fully-hadronic

*/
//
// Original Author:  Michael Maes
//         Created:  Wed Dec  3 12:07:13 CET 2009
//
//

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>
#include <cstdlib>
#include <string>

class PythiaFilterTTBar : public edm::global::EDFilter<> {
public:
  explicit PythiaFilterTTBar(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const unsigned int decayType_;
  const unsigned int leptonFlavour_;
};

PythiaFilterTTBar::PythiaFilterTTBar(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      decayType_(iConfig.getUntrackedParameter("decayType", 1)),
      leptonFlavour_(iConfig.getUntrackedParameter("leptonFlavour", 0)) {}

bool PythiaFilterTTBar::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accept = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  unsigned int iE = 0, iMu = 0, iTau = 0;

  unsigned int iNuE = 0, iNuMu = 0, iNuTau = 0;

  unsigned int iLep = 0, iNu = 0;

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    int pdgID = (*p)->pdg_id();

    int status = (*p)->status();

    if (status == 3) {
      // count the final state leptons

      if (std::abs(pdgID) == 11)
        iE++;

      if (std::abs(pdgID) == 13)
        iMu++;

      if (std::abs(pdgID) == 15)
        iTau++;

      // count the final state neutrinos

      if (std::abs(pdgID) == 12)
        iNuE++;

      if (std::abs(pdgID) == 14)
        iNuMu++;

      if (std::abs(pdgID) == 16)
        iNuTau++;
    }
  }

  iLep = (iE + iMu + iTau);
  iNu = (iNuE + iNuMu + iNuTau);

  if (decayType_ == 1) {  // semi-leptonic decay

    // l = e,mu,tau

    if (leptonFlavour_ == 0 && iLep == 1 && iNu == 1)
      accept = true;

    // l = e

    else if (leptonFlavour_ == 1 && iE == 1 && iNuE == 1 && iLep == 1 && iNu == 1)
      accept = true;

    // l = mu

    else if (leptonFlavour_ == 2 && iMu == 1 && iNuMu == 1 && iLep == 1 && iNu == 1)
      accept = true;

    // l = tau

    else if (leptonFlavour_ == 3 && iTau == 1 && iNuTau == 1 && iLep == 1 && iNu == 1)
      accept = true;

  }

  else if (decayType_ == 2) {  // di-leptonic decay (inclusive)

    if (iLep == 2 && iNu == 2)
      accept = true;

  }

  else if (decayType_ == 3) {  // fully-hadronic decay

    if (iLep == 0 && iNu == 0)
      accept = true;
  }

  else
    accept = false;

  return accept;
}

DEFINE_FWK_MODULE(PythiaFilterTTBar);
