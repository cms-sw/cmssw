import FWCore.ParameterSet.Config as cms

hltAK4PFPuppiJetCorrectorL3 = cms.EDProducer("LXXXCorrectorProducer",
    #algorithm = cms.string('AK4PFPuppiHLT'),
    algorithm = cms.string('AK4PFPuppi'),
    level = cms.string('L3Absolute')
)
