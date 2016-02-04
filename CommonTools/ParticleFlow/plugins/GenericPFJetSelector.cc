#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/ParticleFlow/interface/GenericPFJetSelectorDefinition.h"

typedef ObjectSelector< pf2pat::GenericPFJetSelectorDefinition ,  reco::PFJetCollection > GenericPFJetSelector;

DEFINE_FWK_MODULE(GenericPFJetSelector);
