import FWCore.ParameterSet.Config as cms

from RecoTracker.MeasurementDet.measurementTrackerEventDefault_cfi import measurementTrackerEventDefault as _measurementTrackerEventDefault

MeasurementTrackerEvent = _measurementTrackerEventDefault.clone(
    badPixelFEDChannelCollectionLabels = ['siPixelDigis'],
)

# This customization will be removed once we have phase2 pixel digis
# Need this line to stop error about missing siPixelDigis
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(MeasurementTrackerEvent, # FIXME
    inactivePixelDetectorLabels = [],
    Phase2TrackerCluster1DProducer = 'siPhase2Clusters',
    stripClusterProducer = ''
)

MeasurementTrackerEventPreSplitting = MeasurementTrackerEvent.clone(
    pixelClusterProducer = 'siPixelClustersPreSplitting'
    )
