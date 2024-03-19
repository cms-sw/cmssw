import FWCore.ParameterSet.Config as cms

hltAK4PFPuppiJetCorrectorL3 = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFPuppiHLT'),
    level = cms.string('L3Absolute')
)
