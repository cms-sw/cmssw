import FWCore.ParameterSet.Config as cms

hltAK4PFCHSJetCorrectorL3 = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L3Absolute')
)
