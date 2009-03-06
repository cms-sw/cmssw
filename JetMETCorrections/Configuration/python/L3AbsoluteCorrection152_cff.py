import FWCore.ParameterSet.Config as cms

L3JetCorrectorIcone5 = cms.ESSource("L3AbsoluteCorrectionService",
    tagName = cms.string('CMSSW_152_L3Absolute_iterativeCone5'),
    label = cms.string('L3AbsoluteJetCorrectorIcone5')
)

L3JetCorrectorIcone7 = cms.ESSource("L3AbsoluteCorrectionService",
    tagName = cms.string('CMSSW_152_L3Absolute_iterativeCone7'),
    label = cms.string('L3AbsoluteJetCorrectorIcone7')
)

L3JetCorrectorMcone5 = cms.ESSource("L3AbsoluteCorrectionService",
    tagName = cms.string('CMSSW_152_L3Absolute_midPointCone5'),
    label = cms.string('L3AbsoluteJetCorrectorMcone5')
)

L3JetCorrectorMcone7 = cms.ESSource("L3AbsoluteCorrectionService",
    tagName = cms.string('CMSSW_152_L3Absolute_midPointCone7'),
    label = cms.string('L3AbsoluteJetCorrectorMcone7')
)

L3JetCorrectorScone5 = cms.ESSource("L3AbsoluteCorrectionService",
    tagName = cms.string('CMSSW_152_L3Absolute_sisCone5'),
    label = cms.string('L3AbsoluteJetCorrectorScone5')
)

L3JetCorrectorScone7 = cms.ESSource("L3AbsoluteCorrectionService",
    tagName = cms.string('CMSSW_152_L3Absolute_sisCone7'),
    label = cms.string('L3AbsoluteJetCorrectorScone7')
)

L3JetCorrectorFKt6 = cms.ESSource("L3AbsoluteCorrectionService",
    tagName = cms.string('CMSSW_152_L3Absolute_fastjet6'),
    label = cms.string('L3AbsoluteJetCorrectorFKt6')
)

L3JetCorrectorFKt4 = cms.ESSource("L3AbsoluteCorrectionService",
    tagName = cms.string('CMSSW_152_L3Absolute_fastjet4'),
    label = cms.string('L3AbsoluteJetCorrectorFKt4')
)

#************** Modules *************************************
L3JetCorJetIcone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("L2JetCorJetIcone5"),
    correctors = cms.vstring('L3AbsoluteJetCorrectorIcone5'),
    alias = cms.untracked.string('L3JetCorJetIcone5')
)

L3JetCorJetIcone7 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("L2JetCorJetIcone7"),
    correctors = cms.vstring('L3AbsoluteJetCorrectorIcone7'),
    alias = cms.untracked.string('L3JetCorJetIcone7')
)

L3JetCorJetMcone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("L2JetCorJetMcone5"),
    correctors = cms.vstring('L3AbsoluteJetCorrectorMcone5'),
    alias = cms.untracked.string('L3JetCorJetMcone5')
)

L3JetCorJetMcone7 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("L2JetCorJetMcone7"),
    correctors = cms.vstring('L3AbsoluteJetCorrectorMcone7'),
    alias = cms.untracked.string('L3JetCorJetMcone7')
)

L3JetCorJetScone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("L2JetCorJetScone5"),
    correctors = cms.vstring('L3AbsoluteJetCorrectorScone5'),
    alias = cms.untracked.string('L3JetCorJetScone5')
)

L3JetCorJetScone7 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("L2JetCorJetScone7"),
    correctors = cms.vstring('L3AbsoluteJetCorrectorScone7'),
    alias = cms.untracked.string('L3JetCorJetScone7')
)

L3JetCorJetFKt6 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("L2JetCorJetFKt6"),
    correctors = cms.vstring('L3AbsoluteJetCorrectorFKt6'),
    alias = cms.untracked.string('L3JetCorJetFKt6')
)

L3JetCorJetFKt4 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("L2JetCorJetFKt4"),
    correctors = cms.vstring('L3AbsoluteJetCorrectorFKt4'),
    alias = cms.untracked.string('L3JetCorJetFKt4')
)


