#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

DEFINE_SEAL_MODULE();

#include "PhysicsTools/UtilAlgos/interface/EventSelector.h"

#include "PhysicsTools/RecoUtils/plugins/CandidateEventSelector.h"

DEFINE_EDM_PLUGIN(EventSelectorFactory, CandidateEventSelector, "CandidateEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, CandidateSEventSelector, "CandidateSEventSelector,");
DEFINE_EDM_PLUGIN(EventSelectorFactory, CandidateSEventVetoSelector, "CandidateSEventVetoSelector");

#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"

namespace configurableAnalysis{
  char Candidate[]="reco::Candidate";
}

typedef ExpressionVariable<reco::Candidate,configurableAnalysis::Candidate> CandidateExpressionVariable;

DEFINE_EDM_PLUGIN(CachingVariableFactory, CandidateExpressionVariable, "CandidateExpressionVariable");
