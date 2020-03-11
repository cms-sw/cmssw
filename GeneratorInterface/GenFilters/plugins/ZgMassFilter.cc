// -*- C++ -*-
//
// Package:    ZgMassFilter
// Class:      ZgMassFilter
//
/*

 Description: filter events based on the Pythia particle information

 Implementation: inherits from generic EDFilter

*/
//
// Original Author:  Alexey Ferapontov
//         Created:  Thu July 26 11:57:54 CDT 2012
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

#include "TLorentzVector.h"

#include <cmath>
#include <cstdlib>
#include <vector>

class ZgMassFilter : public edm::global::EDFilter<> {
public:
  explicit ZgMassFilter(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const double minDileptonMass;
  const double minZgMass;
};

using namespace edm;
using namespace std;

ZgMassFilter::ZgMassFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          iConfig.getUntrackedParameter("moduleLabel", edm::InputTag("generator", "unsmeared")))),
      minDileptonMass(iConfig.getUntrackedParameter("MinDileptonMass", 0.)),
      minZgMass(iConfig.getUntrackedParameter("MinZgMass", 0.)) {}

bool ZgMassFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  vector<TLorentzVector> Lepton;
  Lepton.clear();
  vector<TLorentzVector> Photon;
  Photon.clear();

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if ((*p)->status() == 1 && (abs((*p)->pdg_id()) == 11 || abs((*p)->pdg_id()) == 13 || abs((*p)->pdg_id()) == 15)) {
      TLorentzVector LeptP((*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e());
      Lepton.push_back(LeptP);
    }
    if (abs((*p)->pdg_id()) == 22 && (*p)->status() == 1) {
      TLorentzVector PhotP((*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e());
      Photon.push_back(PhotP);
    }
  }

  if (Lepton.size() > 1 && (Lepton[0] + Lepton[1]).M() > minDileptonMass) {
    if ((Lepton[0] + Lepton[1] + Photon[0]).M() > minZgMass) {
      accepted = true;
    }
  }

  return accepted;
}

DEFINE_FWK_MODULE(ZgMassFilter);
