import FWCore.ParameterSet.Config as cms

# Define CHS correctors
ak5PFchsL1Fastjet = cms.ESProducer(
     'L1FastjetCorrectionESProducer',
     level       = cms.string('L1FastJet'),
     algorithm   = cms.string('AK5PFchs'),
     srcRho      = cms.InputTag('kt6PFJets','rho')
)
ak5PFchsL2Relative = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK5PFchs')
)
ak5PFchsL3Absolute = cms.ESProducer(
     'LXXXCorrectionESProducer',
     level     = cms.string('L3Absolute'),
     algorithm = cms.string('AK5PFchs')
)
ak5PFJetschsL1FastL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFchsL1Fastjet','ak5PFchsL2Relative', 'ak5PFchsL3Absolute')
)

ak5PFchsL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFchsL2Relative', 'ak5PFchsL3Absolute')
    )

