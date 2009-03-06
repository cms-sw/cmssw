import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
MCJetCorrectorIcone5 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_iterativeCone5'),
    label = cms.string('MCJetCorrectorIcone5')
)

MCJetCorrectorScone5 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_sisCone5'),
    label = cms.string('MCJetCorrectorScone5')
)

MCJetCorrectorScone7 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_sisCone7'),
    label = cms.string('MCJetCorrectorScone7')
)

MCJetCorrectorktjet6 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_fastjet6'),
    label = cms.string('MCJetCorrectorktjet6')
)

MCJetCorrectorktjet4 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_fastjet4'),
    label = cms.string('MCJetCorrectorktjet4')
)

MCJetCorJetIcone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('MCJetCorrectorIcone5'),
    alias = cms.untracked.string('MCJetCorJetIcone5')
)

MCJetCorJetktjet4 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("kt4CaloJets"),
    correctors = cms.vstring('MCJetCorrectorktjet4'),
    alias = cms.untracked.string('MCJetCorJetktjet4')
)

MCJetCorJetktjet6 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("kt6CaloJets"),
    correctors = cms.vstring('MCJetCorrectorktjet6'),
    alias = cms.untracked.string('MCJetCorJetktjet6')
)

MCJetCorJetScone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("sisCone5CaloJets"),
    correctors = cms.vstring('MCJetCorrectorScone5'),
    alias = cms.untracked.string('MCJetCorJetScone5')
)

MCJetCorJetScone7 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("sisCone7CaloJets"),
    correctors = cms.vstring('MCJetCorrectorScone7'),
    alias = cms.untracked.string('MCJetCorJetScone7')
)

MCJetCorrections = cms.Sequence(MCJetCorJetIcone5*MCJetCorJetktjet4*MCJetCorJetktjet6*MCJetCorJetScone5*MCJetCorJetScone7)

