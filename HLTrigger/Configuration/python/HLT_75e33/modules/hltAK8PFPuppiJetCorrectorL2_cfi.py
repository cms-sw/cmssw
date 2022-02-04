import FWCore.ParameterSet.Config as cms

hltAK8PFPuppiJetCorrectorL2 = cms.EDProducer("LXXXCorrectorProducer",
    #algorithm = cms.string('AK8PFPuppiHLT'),
    algorithm = cms.string('AK8PFPuppi'),
    level = cms.string('L2Relative')
)
