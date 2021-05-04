// -*- C++ -*-
//
// Package:    PythiaFilterHT
// Class:      PythiaFilterHT
//
/**\class PythiaFilterHT PythiaFilterHT.cc IOMC/PythiaFilterHT/src/PythiaFilterHT.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alejandro Gomez Espinosa
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

class PythiaFilterHT : public edm::global::EDFilter<> {
public:
  explicit PythiaFilterHT(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> label_;
  const double minhtcut;
  const int motherID;
};

using namespace std;

PythiaFilterHT::PythiaFilterHT(const edm::ParameterSet& iConfig)
    : label_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      minhtcut(iConfig.getUntrackedParameter("MinHT", 0.)),
      motherID(iConfig.getUntrackedParameter("MotherID", 0)) {}

bool PythiaFilterHT::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(label_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  double HT = 0;

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if (((*p)->status() == 23) && ((abs((*p)->pdg_id()) < 6) || ((*p)->pdg_id() == 21))) {
      if (motherID == 0) {
        HT += (*p)->momentum().perp();
      } else {
        HepMC::GenParticle* mother = (*((*p)->production_vertex()->particles_in_const_begin()));
        if (abs(mother->pdg_id()) == abs(motherID)) {
          HT += (*p)->momentum().perp();
        }
      }
    }
  }
  if (HT > minhtcut)
    accepted = true;
  return accepted;
}

DEFINE_FWK_MODULE(PythiaFilterHT);
