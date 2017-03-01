import FWCore.ParameterSet.Config as cms

from Configuration.Geometry.GeometryRecoDB_cff import *
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
from Configuration.StandardSequences.MagneticField_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *

from RecoLocalTracker.SiStripRecHitConverter.StripCPEgeometric_cfi import *
TTRHBuilderGeometricAndTemplate = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
StripCPE = cms.string('StripCPEfromTrackAngle'), # cms.string('StripCPEgeometric'),
#StripCPE = cms.string('StripCPEgeometric'),
ComponentName = cms.string('WithGeometricAndTemplate'),
PixelCPE = cms.string('PixelCPEGeneric'),
#PixelCPE = cms.string('PixelCPETemplateReco'),
Matcher = cms.string('StandardMatcher'),
ComputeCoarseLocalPositionFromDisk = cms.bool(False)
)

from RecoTracker.TrackProducer.TrackRefitters_cff import *
TrackRefitterForApeEstimator = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
	src = "MuSkim",
	TrajectoryInEvent = True,
	TTRHBuilder = "WithGeometricAndTemplate",
	NavigationSchool = ''
)

TrackRefitterHighPurityForApeEstimator = TrackRefitterForApeEstimator.clone(
    src = 'HighPuritySelector'
)


## FILTER for high purity tracks
import Alignment.APEEstimation.AlignmentTrackSelector_cff
HighPuritySelector = Alignment.APEEstimation.AlignmentTrackSelector_cff.HighPuritySelector
HighPuritySelector.src = 'MuSkim'



## SEQUENCE

RefitterHighPuritySequence = cms.Sequence(
    offlineBeamSpot*
    HighPuritySelector*
    TrackRefitterForApeEstimator
)



