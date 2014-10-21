import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
pfMet = cms.EDProducer(
    "PFMETProducer",
    src = cms.InputTag("particleFlow"),
    alias = cms.string('PFMET'),
    globalThreshold = cms.double(0.0),
    calculateSignificance = cms.bool(False),
    )
##____________________________________________________________________________||
