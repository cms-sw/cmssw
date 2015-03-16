import FWCore.ParameterSet.Config as cms

import RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi
MeasurementTrackerEvent = RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi.MeasurementTrackerEvent.clone(
    pixelClusterProducer = '',
    stripClusterProducer = '',
    inactivePixelDetectorLabels = cms.VInputTag(),
    inactiveStripDetectorLabels = cms.VInputTag(),
    switchOffPixelsIfEmpty = False
)
