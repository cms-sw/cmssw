import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.looseChi2Estimator_cfi import *
from RecoEgamma.EgammaPhotonProducers.propAlongMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.propOppoMomentumWithMaterialForElectrons_cfi import *
#TrajectoryBuilder
from RecoTracker.CkfPattern.CkfTrajectoryBuilder_cff import *
import RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi
from RecoEgamma.EgammaPhotonProducers.trajectoryFilterForConversions_cfi import TrajectoryFilterForConversions

TrajectoryBuilderForConversions = RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi.CkfTrajectoryBuilder.clone(
    estimator            = 'eleLooseChi2',
    TTRHBuilder          = 'WithTrackAngle',
    updator              = 'KFUpdator',
    propagatorAlong      = 'alongMomElePropagator',
    propagatorOpposite   = 'oppositeToMomElePropagator',
    trajectoryFilter     = dict(refToPSet_ = 'TrajectoryFilterForConversions'),
    maxCand              = 5,
    lostHitPenalty       = 30.,
    intermediateCleaning = True,
    alwaysUseInvalidHits = True,
    seedAs5DHit          = False
)
