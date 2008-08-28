import FWCore.ParameterSet.Config as cms

#################### L2 Source definitions ##################################
L2JetCorrectorIcone5 = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('CSA07_100pb_L2Relative_iterativeCone5'),
    label = cms.string('L2RelativeJetCorrectorIcone5')
)
L2JetCorrectorMcone5 = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('CSA07_100pb_L2Relative_midPointCone5'),
    label = cms.string('L2RelativeJetCorrectorMcone5')
)
L2JetCorrectorMcone7 = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('CSA07_100pb_L2Relative_midPointCone7'),
    label = cms.string('L2RelativeJetCorrectorMcone7')
)
L2JetCorrectorScone5 = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('CSA07_100pb_L2Relative_sisCone5'),
    label = cms.string('L2RelativeJetCorrectorScone5')
)
L2JetCorrectorScone7 = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('CSA07_100pb_L2Relative_sisCone7'),
    label = cms.string('L2RelativeJetCorrectorScone7')
)
L2JetCorrectorKt4 = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('CSA07_100pb_L2Relative_fastjet4'),
    label = cms.string('L2RelativeJetCorrectorKt4')
)
L2JetCorrectorKt6 = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('CSA07_100pb_L2Relative_fastjet6'),
    label = cms.string('L2RelativeJetCorrectorKt6')
)
#################### L3 Source definitions ##################################
L3JetCorrectorIcone5 = cms.ESSource("L3AbsoluteCorrectionService", 
    tagName = cms.string('CSA07_100pb_L3Absolute_iterativeCone5'),
    label = cms.string('L3AbsoluteJetCorrectorIcone5')
)
L3JetCorrectorMcone5 = cms.ESSource("L3AbsoluteCorrectionService", 
    tagName = cms.string('CSA07_100pb_L3Absolute_midPointCone5'),
    label = cms.string('L3AbsoluteJetCorrectorMcone5')
)
L3JetCorrectorMcone7 = cms.ESSource("L3AbsoluteCorrectionService", 
    tagName = cms.string('CSA07_100pb_L3Absolute_midPointCone7'),
    label = cms.string('L3AbsoluteJetCorrectorMcone7')
)
L3JetCorrectorScone5 = cms.ESSource("L3AbsoluteCorrectionService", 
    tagName = cms.string('CSA07_100pb_L3Absolute_sisCone5'),
    label = cms.string('L3AbsoluteJetCorrectorScone5')
)
L3JetCorrectorScone7 = cms.ESSource("L3AbsoluteCorrectionService", 
    tagName = cms.string('CSA07_100pb_L3Absolute_sisCone7'),
    label = cms.string('L3AbsoluteJetCorrectorScone7')
)
L3JetCorrectorKt4 = cms.ESSource("L3AbsoluteCorrectionService", 
    tagName = cms.string('CSA07_100pb_L3Absolute_fastjet4'),
    label = cms.string('L3AbsoluteJetCorrectorKt4')
)
L3JetCorrectorKt6 = cms.ESSource("L3AbsoluteCorrectionService", 
    tagName = cms.string('CSA07_100pb_L3Absolute_fastjet6'),
    label = cms.string('L3AbsoluteJetCorrectorKt6')
)
#################### L2L3 Source definitions ##################################
L2L3JetCorrectorIcone5 = cms.ESSource("JetCorrectionServiceChain",  
    correctors = cms.vstring('L2RelativeJetCorrectorIcone5','L3AbsoluteJetCorrectorIcone5'),
    label = cms.string('L2L3JetCorrectorIcone5') 
)
L2L3JetCorrectorMcone5 = cms.ESSource("JetCorrectionServiceChain",  
    correctors = cms.vstring('L2RelativeJetCorrectorMcone5','L3AbsoluteJetCorrectorMcone5'),
    label = cms.string('L2L3JetCorrectorMcone5') 
)
L2L3JetCorrectorMcone7 = cms.ESSource("JetCorrectionServiceChain",  
    correctors = cms.vstring('L2RelativeJetCorrectorMcone7','L3AbsoluteJetCorrectorMcone7'),
    label = cms.string('L2L3JetCorrectorMcone7') 
) 
L2L3JetCorrectorScone5 = cms.ESSource("JetCorrectionServiceChain",  
    correctors = cms.vstring('L2RelativeJetCorrectorScone5','L3AbsoluteJetCorrectorScone5'),
    label = cms.string('L2L3JetCorrectorScone5') 
)
L2L3JetCorrectorScone7 = cms.ESSource("JetCorrectionServiceChain",  
    correctors = cms.vstring('L2RelativeJetCorrectorScone7','L3AbsoluteJetCorrectorScone7'),
    label = cms.string('L2L3JetCorrectorScone7') 
) 
L2L3JetCorrectorKt4 = cms.ESSource("JetCorrectionServiceChain",  
    correctors = cms.vstring('L2RelativeJetCorrectorKt4','L3AbsoluteJetCorrectorKt4'),
    label = cms.string('L2L3JetCorrectorKt4') 
)
L2L3JetCorrectorKt6 = cms.ESSource("JetCorrectionServiceChain",  
    correctors = cms.vstring('L2RelativeJetCorrectorKt6','L3AbsoluteJetCorrectorKt6'),
    label = cms.string('L2L3JetCorrectorKt6') 
)
#################### L2L3 Module definitions ##################################
L2L3CorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorIcone5')
)
L2L3CorJetMcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("midPointCone5CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorMcone5')
)
L2L3CorJetMcone7 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("midPointCone7CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorMcone7')
)
L2L3CorJetScone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("sisCone5CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorScone5')
)
L2L3CorJetScone7 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("sisCone7CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorScone7')
)
L2L3CorJetKt4 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("kt4CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorKt4')
)
L2L3CorJetKt6 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("kt6CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorKt6')
)
