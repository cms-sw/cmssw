import FWCore.ParameterSet.Config as cms

from RecoTracker.MeasurementDet.measurementTrackerEventDefault_cfi import measurementTrackerEventDefault as _measurementTrackerEventDefault

MeasurementTrackerEvent = _measurementTrackerEventDefault.clone(
    badPixelFEDChannelCollectionLabels = ['siPixelDigis'],
)

# in case of RAW' (approximated SiStrip clusters) 
# take the list of inactive strip labels directly from RAW data
from Configuration.ProcessModifiers.approxSiStripClusters_cff import approxSiStripClusters
approxSiStripClusters.toModify(MeasurementTrackerEvent,
                               inactiveStripDetectorLabels = ["hltSiStripRawToDigi"])

# This customization will be removed once we have phase2 pixel digis
# Need this line to stop error about missing siPixelDigis
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(MeasurementTrackerEvent, # FIXME
    inactivePixelDetectorLabels = [],
    Phase2TrackerCluster1DProducer = 'siPhase2Clusters',
    stripClusterProducer = ''
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(MeasurementTrackerEvent,
    pixelClusterProducer = '',
    stripClusterProducer = '',
    inactivePixelDetectorLabels = [],
    inactiveStripDetectorLabels = [],
    switchOffPixelsIfEmpty = False
)
from Configuration.ProcessModifiers.vectorHits_cff import vectorHits
vectorHits.toModify(MeasurementTrackerEvent,
    vectorHits = "siPhase2VectorHits:accepted",
    vectorHitsRej = "siPhase2VectorHits:rejected",
)

MeasurementTrackerEventPreSplitting = MeasurementTrackerEvent.clone(
    pixelClusterProducer = 'siPixelClustersPreSplitting'
)

# in case of RAW' (approximated SiStrip clusters) 
# take the list of inactive strip labels directly from RAW data 
approxSiStripClusters.toModify(MeasurementTrackerEventPreSplitting,
                               inactiveStripDetectorLabels = ["hltSiStripRawToDigi"])
