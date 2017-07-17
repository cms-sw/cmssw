#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"

namespace configurableAnalysis{
  constexpr char Jet[]="pat::Jet";
  constexpr char Muon[]="pat::Muon";
  constexpr char MET[]="pat::MET";
  constexpr char Electron[]="pat::Electron";
  constexpr char Tau[]="pat::Tau";
  constexpr char Photon[]="pat::Photon";
}

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

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

#include "CommonTools/UtilAlgos/interface/StringCutEventSelector.h"


//single cut object selector
typedef StringCutEventSelector<pat::Jet> patJetEventSelector;
typedef StringCutEventSelector<pat::Muon> patMuonEventSelector;
typedef StringCutEventSelector<pat::MET> patMETEventSelector;
typedef StringCutEventSelector<pat::Electron> patElectronEventSelector;
typedef StringCutEventSelector<pat::Photon> patPhotonEventSelector;
typedef StringCutEventSelector<pat::Tau> patTauEventSelector;

//selector with multiple cuts
typedef StringCutsEventSelector<pat::Jet> patJetSEventSelector;
typedef StringCutsEventSelector<pat::Muon> patMuonSEventSelector;
typedef StringCutsEventSelector<pat::MET> patMETSEventSelector;
typedef StringCutsEventSelector<pat::Electron> patElectronSEventSelector;
typedef StringCutsEventSelector<pat::Photon> patPhotonSEventSelector;
typedef StringCutsEventSelector<pat::Tau> patTauSEventSelector;

//vetoes
typedef StringCutsEventSelector<pat::Jet,false> patJetSEventVetoSelector;
typedef StringCutsEventSelector<pat::Muon,false> patMuonSEventVetoSelector;
typedef StringCutsEventSelector<pat::MET,false> patMETSEventVetoSelector;
typedef StringCutsEventSelector<pat::Electron,false> patElectronSEventVetoSelector;
typedef StringCutsEventSelector<pat::Photon,false> patPhotonSEventVetoSelector;
typedef StringCutsEventSelector<pat::Tau,false> patTauSEventVetoSelector;
//any selector
typedef StringCutEventSelector<pat::Jet,true> patAnyJetEventSelector;
typedef StringCutEventSelector<pat::Muon,true> patAnyMuonEventSelector;
typedef StringCutEventSelector<pat::Electron,true> patAnyElectronEventSelector;

DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patJetEventSelector, "patJetEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patAnyJetEventSelector, "patAnyJetEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patJetSEventSelector, "patJetSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patJetSEventVetoSelector, "patJetSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patMETEventSelector, "patMETEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patMETSEventSelector, "patMETSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patMETSEventVetoSelector, "patMETSEventVeloSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patMuonEventSelector, "patMuonEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patAnyMuonEventSelector, "patAnyMuonEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patMuonSEventSelector, "patMuonSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patMuonSEventVetoSelector, "patMuonSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patElectronEventSelector, "patElectronEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patAnyElectronEventSelector, "patAnyElectronEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patElectronSEventSelector, "patElectronSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patElectronSEventVetoSelector, "patElectronSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patPhotonEventSelector, "patPhotonEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patPhotonSEventSelector, "patPhotonSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patPhotonSEventVetoSelector, "patPhotonSEventVetoSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patTauEventSelector, "patTauEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patTauSEventSelector, "patTauSEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, patTauSEventVetoSelector, "patTauSEventVetoSelector");

#include "PhysicsTools/PatUtils/interface/RazorComputer.h"
DEFINE_EDM_PLUGIN(CachingVariableFactory, RazorBox, "RazorBox");
DEFINE_EDM_PLUGIN(VariableComputerFactory, RazorComputer, "RazorComputer");

