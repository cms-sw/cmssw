import FWCore.ParameterSet.Config as cms

hltAK8PFCHSJetCorrectorL3 = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK8PFchs'),
    level = cms.string('L3Absolute')
)
