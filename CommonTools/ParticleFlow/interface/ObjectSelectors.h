#ifndef CommonTools_ParticleFlow_ObjectSelectors
#define CommonTools_ParticleFlow_ObjectSelectors

#include "CommonTools/ParticleFlow/interface/ObjectSelector.h"
#include "CommonTools/ParticleFlow/interface/PtMinPFCandidateSelectorDefinition.h"
#include "CommonTools/ParticleFlow/interface/PdgIdPFCandidateSelectorDefinition.h"
#include "CommonTools/ParticleFlow/interface/IsolatedPFCandidateSelectorDefinition.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"


namespace ipf2pat {
  typedef ObjectSelector< pf2pat::PtMinPFCandidateSelectorDefinition, reco::PFCandidateCollection > PtMinPFCandidateSelector;
  typedef ObjectSelector< pf2pat::PdgIdPFCandidateSelectorDefinition, reco::PFCandidateCollection > PdgIdPFCandidateSelector;
  typedef ObjectSelector< pf2pat::IsolatedPFCandidateSelectorDefinition, reco::PFCandidateCollection > IsolatedPFCandidateSelector;
}

#endif
