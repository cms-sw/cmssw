import FWCore.ParameterSet.Config as cms

hltAK4PFPuppiJetCorrectorL2 = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFPuppiHLT'),
    level = cms.string('L2Relative')
)
