import FWCore.ParameterSet.Config as cms

MeasurementTrackerEvent = cms.EDProducer("MeasurementTrackerEventProducer",
    measurementTracker = cms.string(''),

    skipClusters = cms.InputTag(""),

    pixelClusterProducer = cms.string('siPixelClusters'),
    stripClusterProducer = cms.string('siStripClusters'),

    # One or more DetIdCollections of modules to mask on the fly for a given event
    inactivePixelDetectorLabels = cms.VInputTag(cms.InputTag('siPixelDigis')),
    inactiveStripDetectorLabels = cms.VInputTag(cms.InputTag('siStripDigis')),
    switchOffPixelsIfEmpty = cms.bool(True), # let's keep it like this, for cosmics                                    
)


