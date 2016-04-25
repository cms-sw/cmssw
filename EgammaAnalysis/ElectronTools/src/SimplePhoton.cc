#ifndef SimplePhoton_STANDALONE
#include "EgammaAnalysis/ElectronTools/interface/SimplePhoton.h"

SimplePhoton::SimplePhoton(const reco::Photon &in, unsigned int runNumber, bool isMC) :
  run_(runNumber),
  eClass_(-1), 
  r9_(in.full5x5_r9()),
  scEnergy_(in.superCluster()->rawEnergy() + in.isEB() ? 0 : in.superCluster()->preshowerEnergy()), 
  scEnergyError_(-999.),  // FIXME???
  regEnergy_(in.getCorrectedEnergy(reco::Photon::P4type::regression2)), 
  regEnergyError_(in.getCorrectedEnergyError(reco::Photon::P4type::regression2)), 
  eta_(in.superCluster()->eta()), 
  isEB_(in.isEB()), 
  isMC_(isMC), 
  newEnergy_(regEnergy_), 
  newEnergyError_(regEnergyError_),
  scale_(1.0), smearing_(0.0)
{}

void SimplePhoton::writeTo(reco::Photon & out) const {
  out.setCorrectedEnergy(reco::Photon::P4type::regression2, getNewEnergy(), getNewEnergyError(), true);     
}
#endif
