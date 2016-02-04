#ifndef TauTagTools_PFCandidateMergerBase
#define TauTagTools_PFCandidateMergerBase 

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PFCandidateMergerBase 
{
 public:
  PFCandidateMergerBase(const edm::ParameterSet&);
  PFCandidateMergerBase();

  virtual ~PFCandidateMergerBase()=0;

  virtual std::vector<reco::PFCandidateRefVector> mergeCandidates(const reco::PFCandidateRefVector&) =0;

};

#endif


