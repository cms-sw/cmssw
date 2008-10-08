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
typedef StringCutsEventSelector<pat::Muon,false> MuonSEventVetoSelector;
typedef StringCutEventSelector<pat::MET> METEventSelector;
typedef StringCutsEventSelector<pat::MET> METSEventSelector;
typedef StringCutsEventSelector<pat::MET,false> METSEventVetoSelector;
typedef StringCutEventSelector<pat::Electron> ElectronEventSelector;
typedef StringCutsEventSelector<pat::Electron> ElectronSEventSelector;
typedef StringCutsEventSelector<pat::Electron,false> ElectronSEventVetoSelector;
typedef StringCutEventSelector<pat::Photon> PhotonEventSelector;
typedef StringCutsEventSelector<pat::Photon> PhotonSEventSelector;
typedef StringCutsEventSelector<pat::Photon,false> PhotonSEventVetoSelector;
typedef StringCutEventSelector<pat::Tau> TauEventSelector;
typedef StringCutsEventSelector<pat::Tau> TauSEventSelector;
typedef StringCutsEventSelector<pat::Tau,false> TauSEventVetoSelector;

DEFINE_EDM_PLUGIN(EventSelectorFactory, JetEventSelector, "JetEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, JetSEventSelector, "JetSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, JetSEventVetoSelector, "JetSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, METEventSelector, "METEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, METSEventSelector, "METSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, METSEventVetoSelector, "METSEventVeloSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, MuonEventSelector, "MuonEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, MuonSEventSelector, "MuonSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, MuonSEventVetoSelector, "MuonSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, ElectronEventSelector, "ElectronEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, ElectronSEventSelector, "ElectronSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, ElectronSEventVetoSelector, "ElectronSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, PhotonEventSelector, "PhotonEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, PhotonSEventSelector, "PhotonSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, PhotonSEventVetoSelector, "PhotonSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, TauEventSelector, "TauEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, TauSEventSelector, "TauSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, TauSEventVetoSelector, "TauSEventVetoSelector");

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
typedef ExpressionVariable<pat::Electron,configurableAnalysis::Electron> ElectronExpressionVariable;
typedef ExpressionVariable<pat::Photon,configurableAnalysis::Photon> PhotonExpressionVariable;
typedef ExpressionVariable<pat::Tau,configurableAnalysis::Tau> TauExpressionVariable;

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

