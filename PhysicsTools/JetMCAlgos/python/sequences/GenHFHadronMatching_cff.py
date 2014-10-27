import FWCore.ParameterSet.Config as cms

# Supplies PDG ID to real name resolution of MC particles
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# Ghost particle collection
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import selectedHadronsAndPartons
selectedHadronsAndPartons = selectedHadronsAndPartons.clone()

# Flavour info: jet collection with all associated ghosts
from PhysicsTools.JetMCAlgos.AK5PFJetsMCFlavourInfos_cfi import ak5JetFlavourInfos
genJetFlavourInfos = ak5JetFlavourInfos.clone(
    leptons = cms.InputTag("selectedHadronsAndPartons","leptons")
)


from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cfi import matchGenHFHadron


# Configuration for matching B-hadrons ================================================================
matchGenBHadron = matchGenHFHadron.clone()
matchGenBHadron.flavour = 5
genBHadronMatchingSequence = cms.Sequence(
    selectedHadronsAndPartons * 
    genJetFlavourInfos * 
    matchGenBHadron 
)


# Configuration for matching C-hadrons =================================================================
matchGenCHadron = matchGenHFHadron.clone()
matchGenCHadron.flavour = 4
genCHadronMatchingSequence = cms.Sequence(
    selectedHadronsAndPartons * 
    genJetFlavourInfos * 
    matchGenCHadron 
)


# Configuration for matching B- and C-hadrons ==========================================================
genBCHadronMatchingSequence = cms.Sequence(
    selectedHadronsAndPartons * 
    genJetFlavourInfos * 
    matchGenBHadron *
    matchGenCHadron
)


