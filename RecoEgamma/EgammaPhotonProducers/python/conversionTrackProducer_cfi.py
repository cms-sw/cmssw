import FWCore.ParameterSet.Config as cms

conversionTrackProducer = cms.EDProducer("ConversionTrackProducer",
    #input collection of tracks or gsf tracks
    TrackProducer = cms.string(''),
    #control which flags are set in output collection
    setTrackerOnly = cms.bool(False),
    setArbitratedEcalSeeded = cms.bool(False),
    setArbitratedMerged = cms.bool(True),
)
