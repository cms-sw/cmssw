import FWCore.ParameterSet.Config as cms

# probably this has to be moved
# why is FastSim actually using this?
from FastSimulation.ParticlePropagator.MagneticFieldMapESProducer_cfi import *

# get the track fitters, consider moving and renaming
from FastSimulation.Tracking.GSTrackFinalFitCommon_cff import *

# we need this stuff to prevent MeasurementTrackerEvent to crash
from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cff import *

# services needed by tracking
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import TransientTrackBuilderESProducer
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cfi import navigationSchoolESProducer

from RecoVertex.BeamSpotProducer.BeamSpot_cff import offlineBeamSpot
from FastSimulation.TrackingRecHitProducer.SiTrackerGaussianSmearingRecHitConverter_cfi import siTrackerGaussianSmearingRecHits


from FastSimulation.Tracking.IterativeTracking_cff import *
from TrackingTools.TrackFitters.TrackFitters_cff import *

fastTkReconstruction = cms.Sequence(
    offlineBeamSpot
    * siTrackerGaussianSmearingRecHits
    * iterTracking
    )
