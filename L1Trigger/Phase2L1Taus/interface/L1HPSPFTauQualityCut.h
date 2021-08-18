#ifndef L1Trigger_Phase2L1Taus_L1HPSPFTauQualityCut_h
#define L1Trigger_Phase2L1Taus_L1HPSPFTauQualityCut_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"         // edm::ParameterSet
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"  // l1t::PFCandidate
#include <string>                                               // std::string
#include <vector>                                               // std::vector

class L1HPSPFTauQualityCut {
public:
  /// constructor
  L1HPSPFTauQualityCut(const edm::ParameterSet& cfg);

  /// destructor
  ~L1HPSPFTauQualityCut() = default;

  /// returns true (false) if PFCandidate passes (fails) quality cuts
  bool operator()(const l1t::PFCandidate& pfCand, float_t primaryVertexZ) const;

  /// accessor functions
  l1t::PFCandidate::ParticleType pfCandType() const;
  enum { kDisabled, kEnabledPrimary, kEnabledPileup };
  int dzCut() const;
  float_t minPt() const;
  float_t maxDz() const;

private:
  l1t::PFCandidate::ParticleType pfCandType_;

  int dzCut_;  // flag to invert dz cut in order to compute charged isolation from pileup for delta-beta corrections

  float_t minPt_;
  float_t maxDz_;

  bool debug_;
};

std::vector<L1HPSPFTauQualityCut> readL1PFTauQualityCuts(const edm::ParameterSet& cfg,
                                                         const std::string& dzCut,
                                                         bool debug = false);

bool isSelected(const std::vector<L1HPSPFTauQualityCut>& qualityCuts,
                const l1t::PFCandidate& pfCand,
                float_t primaryVertexZ);

#endif
