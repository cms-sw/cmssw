import FWCore.ParameterSet.Config as cms


# Flavour info: particle collection containing ghosts that will be injected into jets
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
selectedHadronsAndPartons = selectedHadronsAndPartons.clone()

# Flavour info: jet collection with all associated ghosts
from PhysicsTools.JetMCAlgos.AK5PFJetsMCFlavourInfos_cfi import ak5JetFlavourInfos
genJetFlavourPlusLeptonInfos = ak5JetFlavourInfos.clone(
    leptons = cms.InputTag("selectedHadronsAndPartons","leptons")
)


# Importing basic configuration for matching heavy flavour hadrons
from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cfi import matchGenHFHadron


# Configuration for matching B-hadrons ================================================================
matchGenBHadron = matchGenHFHadron.clone(
    jetFlavourInfos = cms.InputTag("genJetFlavourPlusLeptonInfos"),
    flavour = 5
)
matchGenBHadronSequence = cms.Sequence( selectedHadronsAndPartons * genJetFlavourPlusLeptonInfos * matchGenBHadron )


# Configuration for matching C-hadrons =================================================================
matchGenCHadron = matchGenHFHadron.clone(
    jetFlavourInfos = cms.InputTag("genJetFlavourPlusLeptonInfos"),
    flavour = 4
)
matchGenCHadronSequence = cms.Sequence( selectedHadronsAndPartons * genJetFlavourPlusLeptonInfos * matchGenCHadron )

# Configuration for matching both B-hadrons and C-hadrons
matchGenBCHadronSequence = cms.Sequence( selectedHadronsAndPartons * genJetFlavourPlusLeptonInfos * matchGenBHadron * matchGenCHadron )

