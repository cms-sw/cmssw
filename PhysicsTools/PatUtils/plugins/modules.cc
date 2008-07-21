#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

DEFINE_SEAL_MODULE();

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"


#include "PhysicsTools/UtilAlgos/interface/EventSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutEventSelector.h"

//#include "Workspace/ConfigurableAnalysis/plugins/JetMetDphiEventSelector.h"
//#include "Workspace/ConfigurableAnalysis/plugins/JetJetDphiEventSelector.h"
//#include "Workspace/ConfigurableAnalysis/plugins/CandidateEventSelector.h"
typedef StringCutEventSelector<pat::Jet> JetEventSelector;
typedef StringCutsEventSelector<pat::Jet> JetSEventSelector;
typedef StringCutsEventSelector<pat::Jet,false> JetSEventVetoSelector;
typedef StringCutEventSelector<pat::Muon> MuonEventSelector;
typedef StringCutsEventSelector<pat::Muon> MuonSEventSelector;
typedef StringCutEventSelector<pat::MET> METEventSelector;
typedef StringCutsEventSelector<pat::MET> METSEventSelector;

//DEFINE_EDM_PLUGIN(EventSelectorFactory, JetMetDphiEventSelector, "JetMetDphiEventSelector");
//DEFINE_EDM_PLUGIN(EventSelectorFactory, JetJetDphiEventSelector, "JetJetDphiEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, JetEventSelector, "JetEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, JetSEventSelector, "JetSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, METEventSelector, "METEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, METSEventSelector, "METSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, MuonEventSelector, "MuonEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, MuonSEventSelector, "MuonSEventSelector");

#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"

namespace configurableAnalysis{
  char Jet[]="pat::Jet";
  char Muon[]="pat::Muon";
  char MET[]="pat::MET";
  char Electron[]="pat::Electron";
  char Tau[]="pat::Tau";
  char Photon[]="pat::Photon";
}

typedef ExpressionVariable<pat::Jet,configurableAnalysis::Jet> JetExpressionVariable;
typedef ExpressionVariable<pat::MET,configurableAnalysis::MET> METExpressionVariable;
typedef ExpressionVariable<pat::Muon,configurableAnalysis::Muon> MuonExpressionVariable;
typedef ExpressionVariable<pat::Tau,configurableAnalysis::Tau> TauExpressionVariable;
typedef ExpressionVariable<pat::Electron,configurableAnalysis::Electron> ElectronExpressionVariable;
typedef ExpressionVariable<pat::Photon,configurableAnalysis::Photon> PhotonExpressionVariable;

DEFINE_EDM_PLUGIN(CachingVariableFactory, JetExpressionVariable, "JetExpressionVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, METExpressionVariable, "METExpressionVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, MuonExpressionVariable, "MuonExpressionVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, ElectronExpressionVariable, "ElectronExpressionVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, PhotonExpressionVariable, "PhotonExpressionVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, TauExpressionVariable, "TauExpressionVariable");

#include "PhysicsTools/UtilAlgos/interface/TwoObjectCalculator.h"

typedef TwoObjectVariable<pat::Jet,configurableAnalysis::Jet,pat::Muon,configurableAnalysis::Muon, CosDphiCalculator> JetMuonCosDphiVariable;
typedef TwoObjectVariable<pat::Jet,configurableAnalysis::Jet,pat::MET,configurableAnalysis::MET, CosDphiCalculator> JetMETCosDphiVariable;
typedef TwoObjectVariable<pat::Jet,configurableAnalysis::Jet,pat::Jet,configurableAnalysis::Jet, CosDphiCalculator> JetJetCosDphiVariable;

DEFINE_EDM_PLUGIN(CachingVariableFactory, JetMuonCosDphiVariable, "JetMuonCosDphiVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, JetMETCosDphiVariable, "JetMETCosDphiVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, JetJetCosDphiVariable, "JetJetCosDphiVariable");
