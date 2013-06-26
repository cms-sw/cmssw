import FWCore.ParameterSet.Config as cms

scaledJets = cms.EDProducer("JetEnergyShift",
    inputJets            = cms.InputTag("cleanPatJets"),
    inputMETs            = cms.InputTag("patMETs"),
    scaleFactor          = cms.double(1.0),
    jetPTThresholdForMET = cms.double(20.),
    jetEMLimitForMET     = cms.double(0.9)
)
