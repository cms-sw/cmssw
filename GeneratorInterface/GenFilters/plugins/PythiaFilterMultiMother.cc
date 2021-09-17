#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "GeneratorInterface/GenFilters/plugins/MCFilterZboostHelper.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

class PythiaFilterMultiMother : public edm::global::EDFilter<> {
public:
  explicit PythiaFilterMultiMother(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------

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
  const int processID;

  const double betaBoost;
};

using namespace std;

PythiaFilterMultiMother::PythiaFilterMultiMother(const edm::ParameterSet& iConfig)
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
      processID(iConfig.getUntrackedParameter("ProcessID", 0)),
      betaBoost(iConfig.getUntrackedParameter("BetaBoost", 0.)) {}

bool PythiaFilterMultiMother::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
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
        for (std::vector<int>::const_iterator motherID = motherIDs.begin(); motherID != motherIDs.end(); ++motherID) {
          if (status == 0 && *motherID == 0) {
            accepted = true;
          }
          if (status != 0 && *motherID == 0) {
            if ((*p)->status() == status)
              accepted = true;
          }

          HepMC::GenParticle* mother = (*((*p)->production_vertex()->particles_in_const_begin()));

          if (status == 0 && *motherID != 0) {
            if (abs(mother->pdg_id()) == abs(*motherID)) {
              accepted = true;
            }
          }
          if (status != 0 && *motherID != 0) {
            if ((*p)->status() == status && abs(mother->pdg_id()) == abs(*motherID)) {
              accepted = true;
            }
          }
        }
      }
    }
  } else {
    accepted = true;
  }
  return accepted;
}

DEFINE_FWK_MODULE(PythiaFilterMultiMother);
