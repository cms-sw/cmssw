import FWCore.ParameterSet.Config as cms

hltPFMETJetCorrectorL2 = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFchs'),
    level = cms.string('L2Relative')
)
