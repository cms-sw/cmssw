#include "HLTriggerOffline/Tau/interface/MCTauCand.h"

MCTauCand::MCTauCand() {
}

MCTauCand::~MCTauCand() {
}

MCTauCand::MCTauCand(const HepMC::GenParticle & c) :
  HepMC::GenParticle(c) {
}

MCTauCand::MCTauCand(const HepMC::GenParticle & c, int decayM, int nprong) :
  HepMC::GenParticle(c), _decayMode(decayM), _nProng(nprong) {
}

MCTauCand::MCTauCand(const HepMC::GenParticle & c, int decayM, int nprong,
		     CLHEP::HepLorentzVector vP4) :
  HepMC::GenParticle(c), _decayMode(decayM), _nProng(nprong), _visibleP4(vP4)  {
}

