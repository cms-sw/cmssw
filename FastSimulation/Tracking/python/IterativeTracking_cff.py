import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.GeneralTracks_cfi import *

from FastSimulation.Tracking.IterativeInitialStep_cff import *
from FastSimulation.Tracking.IterativeDetachedTripletStep_cff import *
from FastSimulation.Tracking.IterativeLowPtTripletStep_cff import *
from FastSimulation.Tracking.IterativePixelPairStep_cff import *
from FastSimulation.Tracking.IterativeMixedTripletStep_cff import *
from FastSimulation.Tracking.IterativePixelLessStep_cff import *
from FastSimulation.Tracking.IterativeTobTecStep_cff import *

import RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi
MeasurementTrackerEvent = RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi.MeasurementTrackerEvent.clone(
    pixelClusterProducer = '',
    stripClusterProducer = '',
    inactivePixelDetectorLabels = cms.VInputTag(),
    inactiveStripDetectorLabels = cms.VInputTag(),
    switchOffPixelsIfEmpty = False
)
iterTracking = cms.Sequence(
    MeasurementTrackerEvent 
    +InitialStep
    +DetachedTripletStep
    +LowPtTripletStep
    +PixelPairStep
    +MixedTripletStep
    +PixelLessStep
    +TobTecStep
    +generalTracksBeforeMixing)
