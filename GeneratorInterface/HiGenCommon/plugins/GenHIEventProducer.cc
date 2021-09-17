// -*- C++ -*-
//
// Package:    GenHIEventProducer
// Class:      GenHIEventProducer
//
/**\class GenHIEventProducer GenHIEventProducer.cc yetkin/GenHIEventProducer/src/GenHIEventProducer.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Thu Aug 13 08:39:51 EDT 2009
//
//

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "HepMC/HeavyIon.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
using namespace std;

//
// class decleration
//

class GenHIEventProducer : public edm::global::EDProducer<> {
public:
  explicit GenHIEventProducer(const edm::ParameterSet&);
  ~GenHIEventProducer() override = default;

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  const edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct> > hepmcSrc_;
  const edm::ESGetToken<ParticleDataTable, edm::DefaultRecord> pdtToken_;
  const edm::EDPutTokenT<edm::GenHIEvent> putToken_;

  double ptCut_;
  const bool doParticleInfo_;
};

//
// constructors and destructor
//
GenHIEventProducer::GenHIEventProducer(const edm::ParameterSet& iConfig)
    : hepmcSrc_(consumes<CrossingFrame<edm::HepMCProduct> >(iConfig.getParameter<edm::InputTag>("src"))),
      pdtToken_(esConsumes<ParticleDataTable, edm::DefaultRecord>()),
      putToken_(produces<edm::GenHIEvent>()),
      doParticleInfo_(iConfig.getUntrackedParameter<bool>("doParticleInfo", false)) {
  if (doParticleInfo_) {
    ptCut_ = iConfig.getParameter<double>("ptCut");
  }
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void GenHIEventProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  const auto& pdt = iSetup.getData(pdtToken_);

  double b = -1;
  int npart = -1;
  int ncoll = 0;
  int nhard = 0;
  double phi = 0;
  double ecc = -1;

  int nCharged = 0;
  int nChargedMR = 0;
  int nChargedPtCut = 0;    // NchargedPtCut bym
  int nChargedPtCutMR = 0;  // NchargedPtCutMR bym

  double meanPt = 0;
  double meanPtMR = 0;
  double EtMR = 0;       // Normalized of total energy bym
  double TotEnergy = 0;  // Total energy bym

  const auto& hepmc = iEvent.get(hepmcSrc_);
  MixCollection<HepMCProduct> mix(&hepmc);

  if (mix.size() < 1) {
    throw cms::Exception("MatchVtx") << "Mixing has " << mix.size() << " sub-events, should have been at least 1"
                                     << endl;
  }

  const HepMCProduct& hievt = mix.getObject(mix.size() - 1);
  const HepMC::GenEvent* evt = hievt.GetEvent();
  if (doParticleInfo_) {
    HepMC::GenEvent::particle_const_iterator begin = evt->particles_begin();
    HepMC::GenEvent::particle_const_iterator end = evt->particles_end();
    for (HepMC::GenEvent::particle_const_iterator it = begin; it != end; ++it) {
      if ((*it)->status() != 1)
        continue;
      int pdg_id = (*it)->pdg_id();
      const ParticleData* part = pdt.particle(pdg_id);
      int charge = static_cast<int>(part->charge());

      if (charge == 0)
        continue;
      float pt = (*it)->momentum().perp();
      float eta = (*it)->momentum().eta();
      float energy = (*it)->momentum().e();  // energy bym
      //float energy = (*it)->momentum().energy(); // energy bym
      nCharged++;
      meanPt += pt;
      // Get the total energy bym
      if (fabs(eta) < 1.0) {
        TotEnergy += energy;
      }
      if (pt > ptCut_) {
        nChargedPtCut++;
        if (fabs(eta) < 0.5) {
          nChargedPtCutMR++;
        }
      }
      // end bym

      if (fabs(eta) > 0.5)
        continue;
      nChargedMR++;
      meanPtMR += pt;
    }
  }
  const HepMC::HeavyIon* hi = evt->heavy_ion();

  if (hi) {
    ncoll = ncoll + hi->Ncoll();
    nhard = nhard + hi->Ncoll_hard();
    int np = hi->Npart_proj() + hi->Npart_targ();
    if (np >= 0) {
      npart = np;
      b = hi->impact_parameter();
      phi = hi->event_plane_angle();
      ecc = hi->eccentricity();
    }
  }

  // Get the normalized total energy bym
  if (TotEnergy != 0) {
    EtMR = TotEnergy / 2;
  }

  if (nChargedMR != 0) {
    meanPtMR /= nChargedMR;
  }
  if (nCharged != 0) {
    meanPt /= nCharged;
  }

  iEvent.emplace(putToken_,
                 b,
                 npart,
                 ncoll,
                 nhard,
                 phi,
                 ecc,
                 nCharged,
                 nChargedMR,
                 meanPt,
                 meanPtMR,
                 EtMR,
                 nChargedPtCut,
                 nChargedPtCutMR);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GenHIEventProducer);
