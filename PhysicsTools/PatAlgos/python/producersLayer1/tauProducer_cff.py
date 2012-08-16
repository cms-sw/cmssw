import FWCore.ParameterSet.Config as cms

# prepare reco information
from PhysicsTools.PatAlgos.recoLayer0.pfCandidateIsoDepositSelection_cff import *
from PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff import *
from PhysicsTools.PatAlgos.recoLayer0.tauDiscriminators_cff import *
# CV: do **not** load PhysicsTools/PatAlgos/python/recoLayer0/tauJetCorrections_cff
#     in order to avoid triggering FileInPath to SQLlite file
#       CondFormats/JetMETObjects/data/TauJec11_V1.db
#    (which is not included in all _4_2_x/4_3_x/4_4_x CMSSW releases yet)
#from PhysicsTools.PatAlgos.recoLayer0.tauJetCorrections_cff import *

# add PAT specifics
from PhysicsTools.JetMCAlgos.TauGenJets_cfi import *
from PhysicsTools.JetMCAlgos.TauGenJetsDecayModeSelectorAllHadrons_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi import *

# produce object
from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import *

makePatTaus = cms.Sequence(
    # reco pre-production
    patHPSPFTauDiscrimination *
    patPFCandidateIsoDepositSelection *
    patPFTauIsolation *
    #patTauJetCorrections *
    # pat specifics
    tauMatch *
    tauGenJets *
    tauGenJetsSelectorAllHadrons *
    tauGenJetMatch *
    # object production
    patTaus
)
