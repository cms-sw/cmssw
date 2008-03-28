import FWCore.ParameterSet.Config as cms

MCJetCorrectorIcone5 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_iterativeCone5'),
    label = cms.string('MCJetCorrectorIcone5')
)

MCJetCorrectorIcone7 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_iterativeCone7'),
    label = cms.string('MCJetCorrectorIcone7')
)

MCJetCorrectorMcone5 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_midPointCone5'),
    label = cms.string('MCJetCorrectorMcone5')
)

MCJetCorrectorMcone7 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_midPointCone7'),
    label = cms.string('MCJetCorrectorMcone7')
)

MCJetCorrectorScone5 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_sisCone5'),
    label = cms.string('MCJetCorrectorScone5')
)

MCJetCorrectorScone7 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_sisCone7'),
    label = cms.string('MCJetCorrectorScone7')
)

MCJetCorrectorfastjet6 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_fastjet6'),
    label = cms.string('MCJetCorrectorfastjet6')
)

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

MCJetCorJetfastjet6 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("fastjet6CaloJets"),
    correctors = cms.vstring('MCJetCorrectorfastjet6'),
    alias = cms.untracked.string('MCJetCorJetfastjet6')
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

MCJetCorrections = cms.Sequence(MCJetCorJetIcone5*MCJetCorJetIcone7*MCJetCorJetMcone5*MCJetCorJetMcone7*MCJetCorJetfastjet6*MCJetCorJetScone5*MCJetCorJetScone7)

