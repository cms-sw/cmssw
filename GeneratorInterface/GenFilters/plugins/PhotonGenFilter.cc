#include "GeneratorInterface/GenFilters/plugins/PhotonGenFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;

PhotonGenFilter::PhotonGenFilter(const edm::ParameterSet &iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))) {
  // Constructor implementation
  ptMin = iConfig.getUntrackedParameter<double>("MinPt", 20.);
  etaMin = iConfig.getUntrackedParameter<double>("MinEta", -2.4);
  etaMax = iConfig.getUntrackedParameter<double>("MaxEta", 2.4);
  drMin = iConfig.getUntrackedParameter<double>("drMin", 0.1);
  ptThreshold = iConfig.getUntrackedParameter<double>("ptThreshold", 2.);
}

PhotonGenFilter::~PhotonGenFilter() {
  // Destructor implementation
}

bool PhotonGenFilter::filter(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  const HepMC::GenEvent *myGenEvent = evt->GetEvent();
  bool accepted_event = false;

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {  // loop through all particles
    if ((*p)->pdg_id() == 22 && !hasAncestor(*p, [](int x) {
          return x > 30 && x != 2212;
        })) {  // loop through all photons that don't come from hadrons
      if ((*p)->momentum().perp() > ptMin && (*p)->status() == 1 && (*p)->momentum().eta() > etaMin &&
          (*p)->momentum().eta() < etaMax) {  // check if photon passes pt and eta cuts
        bool good_photon = true;
        double phi = (*p)->momentum().phi();
        double eta = (*p)->momentum().eta();
        double pt = (*p)->momentum().perp();
        double frixione_isolation_coefficient = pt / (1 - cos(drMin));
        vector<double> particles_pt, particles_deltar;
        for (HepMC::GenEvent::particle_const_iterator q = myGenEvent->particles_begin();
             q != myGenEvent->particles_end();
             ++q) {        // loop through all particles to compute frixione isolation
          if (&p != &q) {  // don't compare the photon to itself
            if ((*q)->momentum().perp() > ptThreshold && (*q)->pdg_id() != 22 && (*q)->pdg_id() != 12 &&
                (*q)->pdg_id() != 14 && (*q)->pdg_id() != 16 &&
                (*q)->status() == 1) {  // check if particle passes pt and status cuts and is not a neutrino or photon
              double phi2 = (*q)->momentum().phi();
              double deltaphi = fabs(phi - phi2);
              if (deltaphi > M_PI)
                deltaphi = 2. * M_PI - deltaphi;
              double eta2 = (*q)->momentum().eta();
              double deltaeta = fabs(eta - eta2);
              double deltaR = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);
              if (deltaR < drMin)  // check if particle is within drMin of photon, for isolation
              {
                particles_pt.push_back((*q)->momentum().perp());
                particles_deltar.push_back(deltaR);
              }
            }
          }
        }
        //order particles_pt and particles_deltar according to increasing particles_deltar
        for (unsigned int i = 0; i < particles_deltar.size(); i++) {
          for (unsigned int j = i + 1; j < particles_deltar.size(); j++) {
            if (particles_deltar[i] > particles_deltar[j]) {
              double temp = particles_deltar[i];
              particles_deltar[i] = particles_deltar[j];
              particles_deltar[j] = temp;
              temp = particles_pt[i];
              particles_pt[i] = particles_pt[j];
              particles_pt[j] = temp;
            }
          }
        }
        //calculate frixione isolation
        double total_pt = 0;
        for (unsigned int i = 0; i < particles_deltar.size(); i++) {
          total_pt += particles_pt[i];
          if (total_pt > frixione_isolation_coefficient * (1 - cos(particles_deltar[i]))) {
            good_photon =
                false;  // if for some delta R the isolation condition is not satisfied, the photon is not good
            break;
          }
        }
        if (good_photon) {
          if (hasAncestor(*p, [](int x) { return x == 11 || x == 13 || x == 15; }) ||
              hasAncestor(
                  *p,
                  [](int x) { return x == 24 || x == 5; },
                  true,
                  false)) {  // check if photon is from decay and defines the event as acceptable
            accepted_event = true;
          }
          if (!hasAncestor(*p, [](int x) { return x == 11 || x == 13 || x == 15; }) &&
              hasAncestor(
                  *p,
                  [](int x) { return x == 24 || x == 5; },
                  false,
                  true)) {  // check if photon comes from the hard process and discards it, in case it is
            return false;
          }
        }
      }
    }
  }

  // Implementation for event filtering
  return accepted_event;  // Accept if it has found at least one good photon from decay and none from hard process
}

void PhotonGenFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<double>("MaxEta", 2.4);
  desc.addUntracked<double>("MinEta", -2.4);
  desc.addUntracked<double>("MinPt", 20.);
  desc.addUntracked<double>("drMin", 0.1);
  desc.addUntracked<double>("ptThreshold", 2.);

  descriptions.add("PhotonGenFilter", desc);
}

bool PhotonGenFilter::hasAncestor(
    HepMC::GenParticle *particle,
    function<bool(int)> check,
    bool isWorBFromDecayCheck,
    bool isWorBPromptCheck) const {  // function to check if a particle has a certain ancestor
  if (!particle)
    return false;

  HepMC::GenVertex *prodVertex = particle->production_vertex();

  // If the particle doesn't have a production vertex, it has no parents.
  if (!prodVertex)
    return false;

  // Loop over all parents (incoming particles) of the vertex
  for (auto parent = prodVertex->particles_begin(HepMC::parents); parent != prodVertex->particles_end(HepMC::parents);
       ++parent) {
    int pdgId = abs((*parent)->pdg_id());

    // Check if the PDG ID respects a check
    if (check(pdgId)) {
      if (isWorBFromDecayCheck) {
        return hasAncestor(
            *parent,
            [](int x) { return x == 6; },
            false,
            false);  // if the photon has a W or b quark ancestor, check if it comes from a top quark
      } else if (isWorBPromptCheck) {
        return !hasAncestor(
            *parent,
            [](int x) { return x == 6; },
            false,
            false);  // if the photon has a W or b quark ancestor, check that it doesn't come from a top quark decay (therefore it comes form the hard process)
      } else
        return true;
    }

    return hasAncestor(
        *parent, check, isWorBFromDecayCheck, isWorBPromptCheck);  // Recursively check the ancestry of the parent
  }

  return false;
}

DEFINE_FWK_MODULE(PhotonGenFilter);