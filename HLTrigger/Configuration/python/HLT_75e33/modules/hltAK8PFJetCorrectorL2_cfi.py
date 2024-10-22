import FWCore.ParameterSet.Config as cms

hltAK8PFJetCorrectorL2 = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK8PF'),
    level = cms.string('L2Relative')
)
