import FWCore.ParameterSet.Config as cms

ctppsLocalTrackLiteProducer = cms.EDProducer("CTPPSLocalTrackLiteProducer",
    tagSiStripTrack = cms.InputTag("totemRPLocalTrackFitter")
)
