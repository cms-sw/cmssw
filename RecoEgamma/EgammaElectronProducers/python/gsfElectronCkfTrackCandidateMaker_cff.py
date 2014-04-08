import FWCore.ParameterSet.Config as cms

# CKFTrackCandidateMaker
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
egammaCkfTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi
#replace egammaCkfTrackCandidates.TransientInitialStateEstimatorParameters =
#      {
#         string propagatorAlongTISE    = "PropagatorWithMaterial"
#         string propagatorOppositeTISE = "PropagatorWithMaterialOpposite"
#      }	
# TrajectoryBuilder
TrajectoryBuilderForPixelMatchGsfElectrons = RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi.CkfTrajectoryBuilder.clone()
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
# Electron propagators and estimators
# Looser chi2 estimator for electron trajectory building
gsfElectronChi2 = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone()
# "backward" propagator for electrons
from RecoEgamma.EgammaElectronProducers.bwdGsfElectronPropagator_cff import *
# "forward" propagator for electrons
from RecoEgamma.EgammaElectronProducers.fwdGsfElectronPropagator_cff import *

egammaCkfTrackCandidates.src = cms.InputTag('ecalDrivenElectronSeeds')
egammaCkfTrackCandidates.TrajectoryBuilderPSet.refToPSet_ = 'TrajectoryBuilderForPixelMatchGsfElectrons'
egammaCkfTrackCandidates.SeedLabel = cms.InputTag('')
egammaCkfTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
egammaCkfTrackCandidates.NavigationSchool = 'SimpleNavigationSchool'
egammaCkfTrackCandidates.RedundantSeedCleaner = 'CachingSeedCleanerBySharedInput'

TrajectoryBuilderForPixelMatchGsfElectrons.trajectoryFilter.refToPSet_ = 'TrajectoryFilterForPixelMatchGsfElectrons'
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

TrajectoryFilterForPixelMatchGsfElectrons = cms.PSet(
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

