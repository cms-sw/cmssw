/*
This filter allows to select events where a given particle originates *either directly 
(daughter) or indirectly ((grand)^n-daughter)*  from a list of possible ancestors.

It also allows to filter on the immediate daughters of the said particle.

Kinematic selections can also be applied on the particle and its daughters.

The example below shows how to select a jpsi->mumu coming from *any* b hadron (notice
that MotherIDs is just 5, i.e. b-quark) however long the decay chain is.
This includes, for example, B->Jpsi + X as well as B->Psi(2S)(->JPsi) X

process.jpsi_from_bhadron_filter = cms.EDFilter("PythiaFilterMultiAncestor",
    DaughterIDs = cms.untracked.vint32(-13, 13),
    DaughterMaxEtas = cms.untracked.vdouble(3., 3.),
    DaughterMaxPts = cms.untracked.vdouble(100000.0, 100000.0),
    DaughterMinEtas = cms.untracked.vdouble(-2.6, -2.6),
    DaughterMinPts = cms.untracked.vdouble(2.5, 2.5),
    MaxEta = cms.untracked.double(3.0),
    MinEta = cms.untracked.double(-3.0),
    MinPt = cms.untracked.double(6.0),
    MotherIDs = cms.untracked.vint32(5),
    ParticleID = cms.untracked.int32(443)
)
*/

// system include files
#include <memory>
#include <iostream>

// user include files
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

// #include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/GenFilters/plugins/MCFilterZboostHelper.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

using namespace edm;
using namespace std;

namespace edm {
  class HepMCProduct;
}

class PythiaFilterMultiAncestor : public edm::global::EDFilter<> {
public:
  explicit PythiaFilterMultiAncestor(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  bool isAncestor(HepMC::GenParticle* particle, int IDtoMatch) const;

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const int particleID;
  const double minpcut;
  const double maxpcut;
  const double minptcut;
  const double maxptcut;
  const double minetacut;
  const double maxetacut;
  const double minrapcut;
  const double maxrapcut;
  const double minphicut;
  const double maxphicut;

  const int status;
  const std::vector<int> motherIDs;
  const std::vector<int> daughterIDs;
  const std::vector<double> daughterMinPts;
  const std::vector<double> daughterMaxPts;
  const std::vector<double> daughterMinEtas;
  const std::vector<double> daughterMaxEtas;

  const int processID;

  const double betaBoost;
};

PythiaFilterMultiAncestor::PythiaFilterMultiAncestor(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      particleID(iConfig.getUntrackedParameter("ParticleID", 0)),
      minpcut(iConfig.getUntrackedParameter("MinP", 0.)),
      maxpcut(iConfig.getUntrackedParameter("MaxP", 10000.)),
      minptcut(iConfig.getUntrackedParameter("MinPt", 0.)),
      maxptcut(iConfig.getUntrackedParameter("MaxPt", 10000.)),
      minetacut(iConfig.getUntrackedParameter("MinEta", -10.)),
      maxetacut(iConfig.getUntrackedParameter("MaxEta", 10.)),
      minrapcut(iConfig.getUntrackedParameter("MinRapidity", -20.)),
      maxrapcut(iConfig.getUntrackedParameter("MaxRapidity", 20.)),
      minphicut(iConfig.getUntrackedParameter("MinPhi", -3.5)),
      maxphicut(iConfig.getUntrackedParameter("MaxPhi", 3.5)),
      status(iConfig.getUntrackedParameter("Status", 0)),
      motherIDs(iConfig.getUntrackedParameter("MotherIDs", std::vector<int>{0})),
      daughterIDs(iConfig.getUntrackedParameter("DaughterIDs", std::vector<int>{0})),
      daughterMinPts(iConfig.getUntrackedParameter("DaughterMinPts", std::vector<double>{0.})),
      daughterMaxPts(iConfig.getUntrackedParameter("DaughterMaxPts", std::vector<double>{10000.})),
      daughterMinEtas(iConfig.getUntrackedParameter("DaughterMinEtas", std::vector<double>{-10.})),
      daughterMaxEtas(iConfig.getUntrackedParameter("DaughterMaxEtas", std::vector<double>{10.})),
      processID(iConfig.getUntrackedParameter("ProcessID", 0)),
      betaBoost(iConfig.getUntrackedParameter("BetaBoost", 0.)) {
  //now do what ever initialization is needed
}

// ------------ access the full genealogy ---------
bool PythiaFilterMultiAncestor::isAncestor(HepMC::GenParticle* particle, int IDtoMatch) const {
  for (HepMC::GenVertex::particle_iterator ancestor = particle->production_vertex()->particles_begin(HepMC::ancestors);
       ancestor != particle->production_vertex()->particles_end(HepMC::ancestors);
       ++ancestor) {
    // std::cout << __LINE__ << "]\t particle's PDG ID " << particle->pdg_id()
    //                       << " \t particle's ancestor's PDG ID " << (*ancestor)->pdg_id()
    //                       << " \t ID to match " << IDtoMatch << std::endl;

    if (abs((*ancestor)->pdg_id()) == abs(IDtoMatch)) {
      //  std::cout << __LINE__ << "]\t found!" << std::endl;
      return true;
    }
  }

  // std::cout << __LINE__ << "]\t nope, no luck" << std::endl;
  return false;
}

// ------------ method called to produce the data  ------------
bool PythiaFilterMultiAncestor::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  using namespace edm;
  bool accepted = false;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  if (processID == 0 || processID == myGenEvent->signal_process_id()) {
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
         ++p) {
      HepMC::FourVector mom = MCFilterZboostHelper::zboost((*p)->momentum(), betaBoost);
      double rapidity = 0.5 * log((mom.e() + mom.pz()) / (mom.e() - mom.pz()));

      if (abs((*p)->pdg_id()) == particleID && mom.rho() > minpcut && mom.rho() < maxpcut &&
          (*p)->momentum().perp() > minptcut && (*p)->momentum().perp() < maxptcut && mom.eta() > minetacut &&
          mom.eta() < maxetacut && rapidity > minrapcut && rapidity < maxrapcut && (*p)->momentum().phi() > minphicut &&
          (*p)->momentum().phi() < maxphicut) {
        // find the mother
        for (std::vector<int>::const_iterator motherID = motherIDs.begin(); motherID != motherIDs.end(); ++motherID) {
          // check status if no mother's pdgID is specified
          if (status == 0 && *motherID == 0) {
            accepted = true;
          }
          if (status != 0 && *motherID == 0) {
            if ((*p)->status() == status)
              accepted = true;
          }

          // check the mother's pdgID
          if (status == 0 && *motherID != 0) {
            // if (abs(mother->pdg_id()) == abs(*motherID)) {
            if (isAncestor(*p, *motherID)) {
              accepted = true;
            }
          }
          if (status != 0 && *motherID != 0) {
            // if ((*p)->status() == status && abs(mother->pdg_id()) == abs(*motherID)){
            if ((*p)->status() == status && isAncestor(*p, *motherID)) {
              accepted = true;
            }
          }
        }

        // find the daughters
        if (accepted & (!daughterIDs.empty())) {
          // if you got this far it means that the mother was found
          // now let's check the daughters
          // use a counter, if there's enough daughters that match the pdg and kinematic
          // criteria accept the event
          uint good_dau = 0;
          for (HepMC::GenVertex::particle_iterator dau = (*p)->end_vertex()->particles_begin(HepMC::children);
               dau != (*p)->end_vertex()->particles_end(HepMC::children);
               ++dau) {
            for (unsigned int i = 0; i < daughterIDs.size(); ++i) {
              // if a daughter has its pdgID among the desired ones, apply kin cuts on it
              // if it survives, add a notch to the counter
              if ((*dau)->pdg_id() == daughterIDs[i]) {
                if ((*dau)->momentum().perp() < daughterMinPts[i])
                  continue;
                if ((*dau)->momentum().perp() > daughterMaxPts[i])
                  continue;
                if ((*dau)->momentum().eta() < daughterMinEtas[i])
                  continue;
                if ((*dau)->momentum().eta() > daughterMaxEtas[i])
                  continue;
                ++good_dau;
              }
            }
          }
          if (good_dau < daughterIDs.size())
            accepted = false;
        }
      }
      // only need to satisfy the conditions _once_
      if (accepted)
        break;
    }

  } else {
    accepted = true;
  }

  if (accepted) {
    return true;
  } else {
    return false;
  }
}

DEFINE_FWK_MODULE(PythiaFilterMultiAncestor);
