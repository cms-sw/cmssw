#include "RecoParticleFlow/PFRootEvent/interface/METManager.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/GenMET.h"

METManager::METManager() {}

double 
METManager::computePFMET( const reco::PFCandidateCollection& pfCands ) {
  return -1;
}

double METManager::computeGenMET( const reco::GenParticleCollection& genParticles ) {
  return -1;
}
  
