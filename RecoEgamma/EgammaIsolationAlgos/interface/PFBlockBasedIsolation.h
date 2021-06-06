//--------------------------------------------------------------------------------------------------
//
//  PfBlockBasedIsolationCalculator.cc
// Authors: N. Marinelli Univ. of Notre Dame
//--------------------------------------------------------------------------------------------------

#ifndef PFBlockBasedIsolation_H
#define PFBlockBasedIsolation_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco {
  class PFBlockElementCluster;
}

class PFBlockBasedIsolation {
public:
  PFBlockBasedIsolation();

  ~PFBlockBasedIsolation();

  void setup(const edm::ParameterSet& conf);

public:
  std::vector<reco::PFCandidateRef> calculate(math::XYZTLorentzVectorD p4,
                                              const reco::PFCandidateRef pfEGCand,
                                              const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle);

private:
  const reco::PFBlockElementCluster* getHighestEtECALCluster(const reco::PFCandidate& pfCand);
  bool passesCleaningPhoton(const reco::PFCandidateRef& pfCand, const reco::PFCandidateRef& pfEGCand);
  bool passesCleaningNeutralHadron(const reco::PFCandidateRef& pfCand, const reco::PFCandidateRef& pfEGCand);

  bool passesCleaningChargedHadron(const reco::PFCandidateRef& pfCand, const reco::PFCandidateRef& pfEGCand);
  bool elementPassesCleaning(const reco::PFCandidateRef& pfCand, const reco::PFCandidateRef& pfEGCand);

private:
  double coneSize_;
};

#endif
