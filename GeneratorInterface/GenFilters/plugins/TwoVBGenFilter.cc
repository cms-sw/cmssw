
#include "GeneratorInterface/GenFilters/plugins/TwoVBGenFilter.h"
#include <iostream>
using namespace std;

TwoVBGenFilter::TwoVBGenFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  src_ = iConfig.getUntrackedParameter<edm::InputTag>("src", edm::InputTag("generator", "unsmeared"));

  eejj_ = iConfig.getParameter<bool>("eejj");
  enujj_ = iConfig.getParameter<bool>("enujj");

  mumujj_ = iConfig.getParameter<bool>("mumujj");
  munujj_ = iConfig.getParameter<bool>("munujj");

  tautaujj_ = iConfig.getParameter<bool>("tautaujj");
  taunujj_ = iConfig.getParameter<bool>("taunujj");

  nunujj_ = iConfig.getParameter<bool>("nunujj");

  //cout << eejj_ << endl;
}

TwoVBGenFilter::~TwoVBGenFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool TwoVBGenFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  int nj = 0;
  int ne = 0;
  int nnu = 0;
  int nmu = 0;
  int ntau = 0;

  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByLabel(src_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if (abs((*p)->pdg_id()) != 23 && abs((*p)->pdg_id()) != 24)
      continue;  // If it is not Z or W, go to the next particle.

    if ((*p)->end_vertex()) {
      for (HepMC::GenVertex::particle_iterator des = (*p)->end_vertex()->particles_begin(HepMC::children);
           des != (*p)->end_vertex()->particles_end(HepMC::children);
           ++des) {
        const HepMC::GenParticle* theDaughter = *des;
        if (isQuark(theDaughter))
          ++nj;
        else if (isNeutrino(theDaughter))
          ++nnu;
        else if (isElectron(theDaughter))
          ++ne;
        else if (isMuon(theDaughter))
          ++nmu;
        else if (isTau(theDaughter))
          ++ntau;
      }
    }
  }

  if (ne == 2 && nj == 2 && eejj_)
    return true;
  else if (ne == 1 && nj == 2 && nnu == 1 && enujj_)
    return true;
  else if (nmu == 2 && nj == 2 && mumujj_)
    return true;
  else if (nmu == 1 && nj == 2 && nnu == 1 && munujj_)
    return true;
  else if (ntau == 2 && nj == 2 && tautaujj_)
    return true;
  else if (ntau == 1 && nj == 2 && nnu == 1 && taunujj_)
    return true;
  else if (nnu == 2 && nj == 2 && nunujj_)
    return true;
  else
    return false;
}

// ------------ method called once each job just before starting event loop  ------------
void TwoVBGenFilter::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void TwoVBGenFilter::endJob() {}

bool TwoVBGenFilter::isQuark(const HepMC::GenParticle* p) {
  bool result;
  int pdgid = std::abs(p->pdg_id());
  if (pdgid == 1 || pdgid == 2 || pdgid == 3 || pdgid == 4 || pdgid == 5 || pdgid == 6)
    result = true;
  else
    result = false;
  return result;
}

bool TwoVBGenFilter::isNeutrino(const HepMC::GenParticle* p) {
  bool result;
  int pdgid = std::abs(p->pdg_id());
  if (pdgid == 12 || pdgid == 14 || pdgid == 16)
    result = true;
  else
    result = false;
  return result;
}

bool TwoVBGenFilter::isElectron(const HepMC::GenParticle* p) {
  bool result;
  int pdgid = std::abs(p->pdg_id());
  if (pdgid == 11)
    result = true;
  else
    result = false;
  return result;
}

bool TwoVBGenFilter::isMuon(const HepMC::GenParticle* p) {
  bool result;
  int pdgid = std::abs(p->pdg_id());
  if (pdgid == 13)
    result = true;
  else
    result = false;
  return result;
}

bool TwoVBGenFilter::isTau(const HepMC::GenParticle* p) {
  bool result;
  int pdgid = std::abs(p->pdg_id());
  if (pdgid == 15)
    result = true;
  else
    result = false;
  return result;
}
