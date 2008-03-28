import FWCore.ParameterSet.Config as cms

# CKFTrackCandidateMaker
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
egammaCkfTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *
#replace egammaCkfTrackCandidates.TransientInitialStateEstimatorParameters =
#      {
#         string propagatorAlongTISE    = "PropagatorWithMaterial"
#         string propagatorOppositeTISE = "PropagatorWithMaterialOpposite"
#      }	
# TrajectoryBuilder
TrajectoryBuilderForPixelMatchGsfElectrons = copy.deepcopy(CkfTrajectoryBuilder)
import copy
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# Electron propagators and estimators
# Looser chi2 estimator for electron trajectory building
gsfElectronChi2 = copy.deepcopy(Chi2MeasurementEstimator)
# "backward" propagator for electrons
from RecoEgamma.EgammaElectronProducers.bwdGsfElectronPropagator_cff import *
# "forward" propagator for electrons
from RecoEgamma.EgammaElectronProducers.fwdGsfElectronPropagator_cff import *
# TrajectoryFilter
from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
TrajectoryFilterForPixelMatchGsfElectrons = copy.deepcopy(trajectoryFilterESProducer)
egammaCkfTrackCandidates.SeedProducer = 'electronPixelSeeds'
egammaCkfTrackCandidates.TrajectoryBuilder = 'TrajectoryBuilderForPixelMatchGsfElectrons'
egammaCkfTrackCandidates.SeedLabel = ''
egammaCkfTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
egammaCkfTrackCandidates.NavigationSchool = 'SimpleNavigationSchool'
egammaCkfTrackCandidates.RedundantSeedCleaner = 'CachingSeedCleanerBySharedInput'
TrajectoryBuilderForPixelMatchGsfElectrons.ComponentName = 'TrajectoryBuilderForPixelMatchGsfElectrons'
TrajectoryBuilderForPixelMatchGsfElectrons.trajectoryFilterName = 'TrajectoryFilterForPixelMatchGsfElectrons'
TrajectoryBuilderForPixelMatchGsfElectrons.maxCand = 3
TrajectoryBuilderForPixelMatchGsfElectrons.intermediateCleaning = False
TrajectoryBuilderForPixelMatchGsfElectrons.propagatorAlong = 'fwdGsfElectronPropagator'
TrajectoryBuilderForPixelMatchGsfElectrons.propagatorOpposite = 'bwdGsfElectronPropagator'
TrajectoryBuilderForPixelMatchGsfElectrons.estimator = 'gsfElectronChi2'
TrajectoryBuilderForPixelMatchGsfElectrons.MeasurementTrackerName = ''
TrajectoryBuilderForPixelMatchGsfElectrons.lostHitPenalty = 30.
TrajectoryBuilderForPixelMatchGsfElectrons.alwaysUseInvalidHits = True
TrajectoryBuilderForPixelMatchGsfElectrons.TTRHBuilder = 'WithTrackAngle'
TrajectoryBuilderForPixelMatchGsfElectrons.updator = 'KFUpdator'
gsfElectronChi2.ComponentName = 'gsfElectronChi2'
gsfElectronChi2.MaxChi2 = 100000.
gsfElectronChi2.nSigma = 3.
TrajectoryFilterForPixelMatchGsfElectrons.ComponentName = 'TrajectoryFilterForPixelMatchGsfElectrons'
TrajectoryFilterForPixelMatchGsfElectrons.filterPset = cms.PSet(
    chargeSignificance = cms.double(-1.0),
    minPt = cms.double(3.0),
    minHitsMinPt = cms.int32(-1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(1),
    maxNumberOfHits = cms.int32(-1),
    maxConsecLostHits = cms.int32(1),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(5)
)

