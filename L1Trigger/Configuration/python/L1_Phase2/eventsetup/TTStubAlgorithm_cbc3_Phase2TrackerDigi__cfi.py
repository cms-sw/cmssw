import FWCore.ParameterSet.Config as cms

TTStubAlgorithm_cbc3_Phase2TrackerDigi_ = cms.ESProducer("TTStubAlgorithm_cbc3_Phase2TrackerDigi_",
    zMatching2S = cms.bool(True),
    zMatchingPS = cms.bool(True)
)
