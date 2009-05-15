#ifndef __RecoParticleFlow_PFRootEvent_METManager__
#define __RecoParticleFlow_PFRootEvent_METManager__

#include "RecoParticleFlow/Benchmark/interface/GenericBenchmark.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"

class METManager {

 public:
  /// arguments: to initialize the GenericBenchmark
  METManager();
  
  double computePFMET( const reco::PFCandidateCollection& pfCands );

  double computeGenMET( const reco::GenParticleCollection& genParticles );
  
 private:
  /// plots comparing pf and gen MET 
  GenericBenchmark  pfVsGenBenchmark_;

  reco::GenMETCollection genMETs_;
  reco::PFMETCollection  pfMETs_;


};

#endif
