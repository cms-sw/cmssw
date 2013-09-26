import FWCore.ParameterSet.Config as cms

measurementTrackerSiStripRefGetterProducer = cms.EDProducer("MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag("SiStripRawToClustersFacility"),
    measurementTracker = cms.string('MeasurementTrackerOnDemand')

    pixelClusterProducer = cms.string('siPixelClusters'),
    stripClusterProducer = cms.string('measurementTrackerSiStripRefGetterProducer'),
    stripLazyGetterProducer = cms.string('SiStripRawToClustersFacility'),

    inactivePixelDetectorLabels = cms.VInputTag(),
    inactiveStripDetectorLabels = cms.VInputTag(),

    switchOffPixelsIfEmpty = cms.bool(True), # let's keep it like this, for cosmics                                    
)


