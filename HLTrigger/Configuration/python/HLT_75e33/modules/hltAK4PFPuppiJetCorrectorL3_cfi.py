import FWCore.ParameterSet.Config as cms

hltAK4PFPuppiJetCorrectorL3 = cms.EDProducer("LXXXCorrectorProducer",
    algorithm = cms.string('AK4PFPuppiHLT'),
    #algorithm = cms.string('AK4PFPuppi'),
    level = cms.string('L3Absolute')
)
# foo bar baz
# 106Df7d8j0Qa6
# c6WAUqkzMaNGQ
