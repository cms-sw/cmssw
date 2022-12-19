#include "GeneratorInterface/Core/interface/TaggedProtonHepMCFilter.h"

TaggedProtonHepMCFilter::TaggedProtonHepMCFilter(const edm::ParameterSet &iConfig)
    : xiMin_(iConfig.getParameter<double>("xiMin")),
      xiMax_(iConfig.getParameter<double>("xiMax")),
      oneOverbeamEnergy_(2.0 / iConfig.getParameter<double>("comEnergy")),
      nProtons_(iConfig.getParameter<int>("nProtons")) {}

bool TaggedProtonHepMCFilter::filter(const HepMC::GenEvent *evt) {
  // Going through the particle list, and count good protons
  int nGoodProtons = 0;
  for (HepMC::GenEvent::particle_const_iterator particle = evt->particles_begin(); particle != evt->particles_end();
       ++particle) {
    if ((*particle)->pdg_id() == proton_PDGID_ && 1 == (*particle)->status()) {
      HepMC::FourVector p4 = (*particle)->momentum();
      double xi = (1.0 - std::abs(p4.pz()) * oneOverbeamEnergy_);
      if (xi > xiMin_ && xi < xiMax_)
        nGoodProtons++;
      if (nGoodProtons >= nProtons_)
        return true;
    }
  }
  return false;
}
