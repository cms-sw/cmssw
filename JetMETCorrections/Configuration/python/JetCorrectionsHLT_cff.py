import FWCore.ParameterSet.Config as cms

MCJetCorrectorIcone5 = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('HLT_L2Relative'),
    label = cms.string('MCJetCorrectorIcone5')
)

hltMCJetCorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("hltIterativeCone5CaloJets"),
    correctors = cms.vstring('MCJetCorrectorIcone5')
)


