#include "PhysicsTools/JetMCAlgos/plugins/TauGenJetDecayModeSelector.h"

#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"

TauGenJetDecayModeSelectorImp::TauGenJetDecayModeSelectorImp(const edm::ParameterSet& cfg)
{
  selectedTauDecayModes_ = cfg.getParameter<vstring>("select");
}

bool TauGenJetDecayModeSelectorImp::operator()(const reco::GenJet& tauGenJet) const
{
  std::string tauGenJetDecayMode = JetMCTagUtils::genTauDecayMode(tauGenJet);
  for ( vstring::const_iterator selectedTauDecayMode = selectedTauDecayModes_.begin();
	selectedTauDecayMode != selectedTauDecayModes_.end(); ++selectedTauDecayMode ) {
    if ( tauGenJetDecayMode == (*selectedTauDecayMode) ) return true;
  }
  return false;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TauGenJetDecayModeSelector);
