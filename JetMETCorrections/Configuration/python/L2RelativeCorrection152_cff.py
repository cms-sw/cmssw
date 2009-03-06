import FWCore.ParameterSet.Config as cms

L2JetCorrectorIcone5 = cms.ESSource("L2RelativeCorrectionService",
    tagName = cms.string('CMSSW_152_L2Relative_iterativeCone5'),
    label = cms.string('L2RelativeJetCorrectorIcone5')
)

L2JetCorrectorIcone7 = cms.ESSource("L2RelativeCorrectionService",
    tagName = cms.string('CMSSW_152_L2Relative_iterativeCone7'),
    label = cms.string('L2RelativeJetCorrectorIcone7')
)

L2JetCorrectorMcone5 = cms.ESSource("L2RelativeCorrectionService",
    tagName = cms.string('CMSSW_152_L2Relative_midPointCone5'),
    label = cms.string('L2RelativeJetCorrectorMcone5')
)

L2JetCorrectorMcone7 = cms.ESSource("L2RelativeCorrectionService",
    tagName = cms.string('CMSSW_152_L2Relative_midPointCone7'),
    label = cms.string('L2RelativeJetCorrectorMcone7')
)

L2JetCorrectorScone5 = cms.ESSource("L2RelativeCorrectionService",
    tagName = cms.string('CMSSW_152_L2Relative_sisCone5'),
    label = cms.string('L2RelativeJetCorrectorScone5')
)

L2JetCorrectorScone7 = cms.ESSource("L2RelativeCorrectionService",
    tagName = cms.string('CMSSW_152_L2Relative_sisCone7'),
    label = cms.string('L2RelativeJetCorrectorScone7')
)

L2JetCorrectorFKt6 = cms.ESSource("L2RelativeCorrectionService",
    tagName = cms.string('CMSSW_152_L2Relative_fastjet6'),
    label = cms.string('L2RelativeJetCorrectorFKt6')
)

L2JetCorrectorFKt4 = cms.ESSource("L2RelativeCorrectionService",
    tagName = cms.string('CMSSW_152_L2Relative_fastjet4'),
    label = cms.string('L2RelativeJetCorrectorFKt4')
)

#************** Modules ************************************* 
L2JetCorJetIcone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('L2RelativeJetCorrectorIcone5'),
    alias = cms.untracked.string('L2JetCorJetIcone5')
)

L2JetCorJetIcone7 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("iterativeCone7CaloJets"),
    correctors = cms.vstring('L2RelativeJetCorrectorIcone7'),
    alias = cms.untracked.string('L2JetCorJetIcone7')
)

L2JetCorJetMcone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("midPointCone5CaloJets"),
    correctors = cms.vstring('L2RelativeJetCorrectorMcone5'),
    alias = cms.untracked.string('L2JetCorJetMcone5')
)

L2JetCorJetMcone7 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("midPointCone7CaloJets"),
    correctors = cms.vstring('L2RelativeJetCorrectorMcone7'),
    alias = cms.untracked.string('L2JetCorJetMcone7')
)

L2JetCorJetScone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("sisCone5CaloJets"),
    correctors = cms.vstring('L2RelativeJetCorrectorScone5'),
    alias = cms.untracked.string('L2JetCorJetScone5')
)

L2JetCorJetScone7 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("sisCone7CaloJets"),
    correctors = cms.vstring('L2RelativeJetCorrectorScone7'),
    alias = cms.untracked.string('L2JetCorJetScone7')
)

L2JetCorJetFKt6 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("fastjet6CaloJets"),
    correctors = cms.vstring('L2RelativeJetCorrectorFKt6'),
    alias = cms.untracked.string('L2JetCorJetFKt6')
)

L2JetCorJetFKt4 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("fastjet4CaloJets"),
    correctors = cms.vstring('L2RelativeJetCorrectorFKt4'),
    alias = cms.untracked.string('L2JetCorJetFKt4')
)


