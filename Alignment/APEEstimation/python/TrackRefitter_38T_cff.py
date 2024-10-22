import FWCore.ParameterSet.Config as cms

from Configuration.Geometry.GeometryRecoDB_cff import *
from Configuration.AlCa.GlobalTag import GlobalTag
from Configuration.StandardSequences.MagneticField_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *

from RecoTracker.TrackProducer.TrackRefitters_cff import *
TrackRefitterForApeEstimator = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    src = "MuSkim",
    TrajectoryInEvent = True,
    TTRHBuilder = "WithAngleAndTemplate",
    NavigationSchool = ''
)

TrackRefitterHighPurityForApeEstimator = TrackRefitterForApeEstimator.clone(
    src = 'HighPuritySelector'
)


## FILTER for high purity tracks
import Alignment.APEEstimation.AlignmentTrackSelector_cff
HighPuritySelector = Alignment.APEEstimation.AlignmentTrackSelector_cff.HighPuritySelector
HighPuritySelector.src = 'MuSkim'

NoPuritySelector = Alignment.APEEstimation.AlignmentTrackSelector_cff.NoPuritySelector
NoPuritySelector.src = 'MuSkim'

## SEQUENCE

RefitterHighPuritySequence = cms.Sequence(
    offlineBeamSpot*
    HighPuritySelector*
    TrackRefitterForApeEstimator
)

RefitterNoPuritySequence = cms.Sequence(
    offlineBeamSpot*
    NoPuritySelector*
    TrackRefitterForApeEstimator
)



