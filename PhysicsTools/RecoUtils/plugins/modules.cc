#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"



#include "CommonTools/UtilAlgos/interface/EventSelector.h"

#include "PhysicsTools/RecoUtils/plugins/CandidateEventSelector.h"

DEFINE_EDM_PLUGIN(EventSelectorFactory, CandidateEventSelector, "CandidateEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, CandidateSEventSelector, "CandidateSEventSelector,");
DEFINE_EDM_PLUGIN(EventSelectorFactory, CandidateSEventVetoSelector, "CandidateSEventVetoSelector");

#include "PhysicsTools/RecoUtils/plugins/HLTEventSelector.h"
DEFINE_EDM_PLUGIN(EventSelectorFactory, HLTEventSelector, "HLTEventSelector");

#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"

namespace configurableAnalysis{
  char Candidate[]="reco::Candidate";
  char GenParticle[]="reco::GenParticle";
}

#include "DataFormats/Candidate/interface/Candidate.h"
typedef ExpressionVariable<reco::Candidate,configurableAnalysis::Candidate> CandidateExpressionVariable;
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
typedef ExpressionVariable<reco::GenParticle,configurableAnalysis::GenParticle> GenParticleExpressionVariable;

DEFINE_EDM_PLUGIN(CachingVariableFactory, CandidateExpressionVariable, "CandidateExpressionVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, GenParticleExpressionVariable, "GenParticleExpressionVariable");

#include "PhysicsTools/RecoUtils/plugins/TriggerVariables.h"

DEFINE_EDM_PLUGIN(CachingVariableFactory, HLTBitVariable, "HLTBitVariable");
DEFINE_EDM_PLUGIN(VariableComputerFactory, L1BitComputer, "L1BitComputer");
DEFINE_EDM_PLUGIN(VariableComputerFactory, HLTBitComputer, "HLTBitComputer");
