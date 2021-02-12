import FWCore.ParameterSet.Config as cms

hltAK8PFPuppiJetCorrectorL3 = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK8PFPuppiHLT'),
    level = cms.string('L3Absolute')
)
