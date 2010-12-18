import FWCore.ParameterSet.Config as cms

# prepare reco information
from PhysicsTools.PatAlgos.recoLayer0.pfCandidateIsoDepositSelection_cff import *
from PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff import *
#from PhysicsTools.PatAlgos.recoLayer0.tauDiscriminators_cff import *

# add PAT specifics
from PhysicsTools.JetMCAlgos.TauGenJets_cfi import *
from PhysicsTools.JetMCAlgos.TauGenJetsDecayModeSelectorAllHadrons_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi import *

# produce object
from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import *

makePatTaus = cms.Sequence(
    # reco pre-production
    patPFCandidateIsoDepositSelection *
    patPFTauIsolation *
    # pat specifics
    tauMatch *
    tauGenJets *
    tauGenJetsSelectorAllHadrons *
    tauGenJetMatch *
    # object production
    patTaus
    )
