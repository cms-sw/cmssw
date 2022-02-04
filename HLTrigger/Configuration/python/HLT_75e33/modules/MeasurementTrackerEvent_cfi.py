import FWCore.ParameterSet.Config as cms

MeasurementTrackerEvent = cms.EDProducer("MeasurementTrackerEventProducer",
    Phase2TrackerCluster1DProducer = cms.string('siPhase2Clusters'),
    badPixelFEDChannelCollectionLabels = cms.VInputTag("siPixelDigis"),
    inactivePixelDetectorLabels = cms.VInputTag(),
    inactiveStripDetectorLabels = cms.VInputTag("siStripDigis"),
    measurementTracker = cms.string(''),
    mightGet = cms.optional.untracked.vstring,
    pixelCablingMapLabel = cms.string(''),
    pixelClusterProducer = cms.string('siPixelClusters'),
    skipClusters = cms.InputTag(""),
    stripClusterProducer = cms.string(''),
    switchOffPixelsIfEmpty = cms.bool(True)
)
