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
typedef StringCutEventSelector<pat::Jet> patJetEventSelector;
typedef StringCutsEventSelector<pat::Jet> patJetSEventSelector;
typedef StringCutsEventSelector<pat::Jet,false> patJetSEventVetoSelector;
typedef StringCutEventSelector<pat::Muon> patMuonEventSelector;
typedef StringCutsEventSelector<pat::Muon> patMuonSEventSelector;
typedef StringCutsEventSelector<pat::Muon,false> patMuonSEventVetoSelector;
typedef StringCutEventSelector<pat::MET> patMETEventSelector;
typedef StringCutsEventSelector<pat::MET> patMETSEventSelector;
typedef StringCutsEventSelector<pat::MET,false> patMETSEventVetoSelector;
typedef StringCutEventSelector<pat::Electron> patElectronEventSelector;
typedef StringCutsEventSelector<pat::Electron> patElectronSEventSelector;
typedef StringCutsEventSelector<pat::Electron,false> patElectronSEventVetoSelector;
typedef StringCutEventSelector<pat::Photon> patPhotonEventSelector;
typedef StringCutsEventSelector<pat::Photon> patPhotonSEventSelector;
typedef StringCutsEventSelector<pat::Photon,false> patPhotonSEventVetoSelector;
typedef StringCutEventSelector<pat::Tau> patTauEventSelector;
typedef StringCutsEventSelector<pat::Tau> patTauSEventSelector;
typedef StringCutsEventSelector<pat::Tau,false> patTauSEventVetoSelector;

DEFINE_EDM_PLUGIN(EventSelectorFactory, patJetEventSelector, "patJetEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patJetSEventSelector, "patJetSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patJetSEventVetoSelector, "patJetSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patMETEventSelector, "patMETEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patMETSEventSelector, "patMETSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patMETSEventVetoSelector, "patMETSEventVeloSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patMuonEventSelector, "patMuonEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patMuonSEventSelector, "patMuonSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patMuonSEventVetoSelector, "patMuonSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patElectronEventSelector, "patElectronEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patElectronSEventSelector, "patElectronSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patElectronSEventVetoSelector, "patElectronSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patPhotonEventSelector, "patPhotonEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patPhotonSEventSelector, "patPhotonSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patPhotonSEventVetoSelector, "patPhotonSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patTauEventSelector, "patTauEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patTauSEventSelector, "patTauSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactory, patTauSEventVetoSelector, "patTauSEventVetoSelector");

#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"

namespace configurableAnalysis{
  char Jet[]="pat::Jet";
  char Muon[]="pat::Muon";
  char MET[]="pat::MET";
  char Electron[]="pat::Electron";
  char Tau[]="pat::Tau";
  char Photon[]="pat::Photon";
}

typedef ExpressionVariable<pat::Jet,configurableAnalysis::Jet> patJetExpressionVariable;
typedef ExpressionVariable<pat::MET,configurableAnalysis::MET> patMETExpressionVariable;
typedef ExpressionVariable<pat::Muon,configurableAnalysis::Muon> patMuonExpressionVariable;
typedef ExpressionVariable<pat::Electron,configurableAnalysis::Electron> patElectronExpressionVariable;
typedef ExpressionVariable<pat::Photon,configurableAnalysis::Photon> patPhotonExpressionVariable;
typedef ExpressionVariable<pat::Tau,configurableAnalysis::Tau> patTauExpressionVariable;

DEFINE_EDM_PLUGIN(CachingVariableFactory, patJetExpressionVariable, "patJetExpressionVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, patMETExpressionVariable, "patMETExpressionVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, patMuonExpressionVariable, "patMuonExpressionVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, patElectronExpressionVariable, "patElectronExpressionVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, patPhotonExpressionVariable, "patPhotonExpressionVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, patTauExpressionVariable, "patTauExpressionVariable");

#include "PhysicsTools/UtilAlgos/interface/TwoObjectCalculator.h"

typedef TwoObjectVariable<pat::Jet,configurableAnalysis::Jet,pat::Muon,configurableAnalysis::Muon, CosDphiCalculator> patJetpatMuonCosDphiVariable;
typedef TwoObjectVariable<pat::Jet,configurableAnalysis::Jet,pat::MET,configurableAnalysis::MET, CosDphiCalculator> patJetpatMETCosDphiVariable;
typedef TwoObjectVariable<pat::Jet,configurableAnalysis::Jet,pat::Jet,configurableAnalysis::Jet, CosDphiCalculator> patJetpatJetCosDphiVariable;

DEFINE_EDM_PLUGIN(CachingVariableFactory, patJetpatMuonCosDphiVariable, "patJetpatMuonCosDphiVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, patJetpatMETCosDphiVariable, "patJetpatMETCosDphiVariable");
DEFINE_EDM_PLUGIN(CachingVariableFactory, patJetpatJetCosDphiVariable, "patJetpatJetCosDphiVariable");

