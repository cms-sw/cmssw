#ifndef PhysicsTools_PFCandProducer_ObjectSelectors
#define PhysicsTools_PFCandProducer_ObjectSelectors

#include "PhysicsTools/PFCandProducer/interface/ObjectSelector.h"
#include "PhysicsTools/PFCandProducer/interface/PtMinPFCandidateSelectorDefinition.h"
#include "PhysicsTools/PFCandProducer/interface/PdgIdPFCandidateSelectorDefinition.h"
#include "PhysicsTools/PFCandProducer/interface/IsolatedPFCandidateSelectorDefinition.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"


namespace ipf2pat {
  typedef ObjectSelector< pf2pat::PtMinPFCandidateSelectorDefinition, reco::PFCandidateCollection > PtMinPFCandidateSelector;
  typedef ObjectSelector< pf2pat::PdgIdPFCandidateSelectorDefinition, reco::PFCandidateCollection > PdgIdPFCandidateSelector;
  typedef ObjectSelector< pf2pat::IsolatedPFCandidateSelectorDefinition, reco::PFCandidateCollection > IsolatedPFCandidateSelector;
}

#endif
