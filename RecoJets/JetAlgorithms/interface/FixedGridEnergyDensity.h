#ifndef FIXEDGRIDENERGYDENSITY_H
#define FIXEDGRIDENERGYDENSITY_H

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

class FixedGridEnergyDensity {

 public:
  FixedGridEnergyDensity(const reco::PFCandidateCollection *input){pfCandidates=input;}
  ~FixedGridEnergyDensity(){};
  enum EtaRegion {Central,Forward,All};
  float fixedGridRho(EtaRegion etaRegion=Central);
  float fixedGridRho(std::vector<float>& etabins,std::vector<float>& phibins);

 private:
    const reco::PFCandidateCollection *pfCandidates;

};

#endif
