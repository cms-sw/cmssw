import FWCore.ParameterSet.Config as cms

# Jet corrections.
#
from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
# Define the correction services for each algorithm.
#
MCJetCorrectorIcone5 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('Spring07_Iterative_Cone_05'),
    label = cms.string('MCJetCorrectorIcone5')
)

MCJetCorrectorIcone7 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('Spring07_Iterative_Cone_07'),
    label = cms.string('MCJetCorrectorIcone7')
)

MCJetCorrectorMcone5 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('Spring07_Midpoint_Cone_05'),
    label = cms.string('MCJetCorrectorMcone5')
)

MCJetCorrectorMcone7 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('Spring07_Midpoint_Cone_07'),
    label = cms.string('MCJetCorrectorMcone7')
)

MCJetCorrectorFastjet10 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('Spring07_Fastjet_10'),
    label = cms.string('MCJetCorrectorFastjet10')
)

MCJetCorrectorFastjet6 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('Spring07_Fastjet_6'),
    label = cms.string('MCJetCorrectorFastjet6')
)

#   
#   Define the producers of corrected jet collections for each algorithm.
#
MCJetCorJetIcone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('MCJetCorrectorIcone5'),
    alias = cms.untracked.string('MCJetCorJetIcone5')
)

MCJetCorJetIcone7 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("iterativeCone7CaloJets"),
    correctors = cms.vstring('MCJetCorrectorIcone7'),
    alias = cms.untracked.string('MCJetCorJetIcone7')
)

MCJetCorJetMcone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("midPointCone5CaloJets"),
    correctors = cms.vstring('MCJetCorrectorMcone5'),
    alias = cms.untracked.string('MCJetCorJetMcone5')
)

MCJetCorJetMcone7 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("midPointCone7CaloJets"),
    correctors = cms.vstring('MCJetCorrectorMcone7'),
    alias = cms.untracked.string('MCJetCorJetMcone7')
)

MCJetCorJetFastjet10 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("fastjet10CaloJets"),
    correctors = cms.vstring('MCJetCorrectorFastjet10'),
    alias = cms.untracked.string('MCJetCorJetFastjet10')
)

MCJetCorJetFastjet6 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("fastjet6CaloJets"),
    correctors = cms.vstring('MCJetCorrectorFastjet6'),
    alias = cms.untracked.string('MCJetCorJetFastjet6')
)

#
#  Define a sequence to make all corrected jet collections at once.
#
MCJetCorrections = cms.Sequence(MCJetCorJetIcone5*MCJetCorJetIcone7*MCJetCorJetMcone5*MCJetCorJetMcone7*MCJetCorJetFastjet10*MCJetCorJetFastjet6)

