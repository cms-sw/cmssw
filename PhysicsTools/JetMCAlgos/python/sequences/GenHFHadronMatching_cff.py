import FWCore.ParameterSet.Config as cms

# Flavour info: jet collection with all associated ghosts
from PhysicsTools.JetMCAlgos.AK5PFJetsMCFlavourInfos_cfi import ak5JetFlavourInfos
genJetFlavourPlusLeptonInfos = ak5JetFlavourInfos.clone(
    leptons = cms.InputTag("selectedHadronsAndPartons","leptons")
)


from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cfi import matchGenHFHadron


# Configuration for matching B-hadrons ================================================================
matchGenBHadron = matchGenHFHadron.clone(
    jetFlavourInfos = cms.InputTag("genJetFlavourPlusLeptonInfos"),
    flavour = 5
)


# Configuration for matching C-hadrons =================================================================
matchGenCHadron = matchGenHFHadron.clone(
    jetFlavourInfos = cms.InputTag("genJetFlavourPlusLeptonInfos"),
    flavour = 4
)

