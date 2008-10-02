import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.elecPreId_cff import *
from RecoParticleFlow.PFTracking.gsfSeedClean_cfi import *
from TrackingTools.GsfTracking.CkfElectronCandidates_cff import *
from TrackingTools.GsfTracking.GsfElectrons_cff import *
from RecoParticleFlow.PFTracking.pfNuclear_cfi import *
from RecoParticleFlow.PFTracking.pfV0_cfi import *
from RecoEgamma.EgammaElectronProducers.gsfElectronCkfTrackCandidateMaker_cff import *
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
gsfElCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import TrackingTools.GsfTracking.GsfElectronFit_cfi
gsfPFtracks = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
particleFlowTrack = cms.Sequence(elecPreId*gsfSeedclean*gsfElCandidates*gsfPFtracks*pfTrackElec)
particleFlowTrackWithNuclear = cms.Sequence(elecPreId*gsfSeedclean*gsfElCandidates*gsfPFtracks*pfTrackElec*pfNuclear)
particleFlowTrackWithV0 = cms.Sequence(elecPreId*gsfSeedclean*gsfElCandidates*gsfPFtracks*pfTrackElec*pfV0)
gsfElCandidates.TrajectoryBuilder = 'TrajectoryBuilderForElectronsinJets'
gsfElCandidates.SeedProducer = 'gsfSeedclean'
gsfElCandidates.SeedLabel = ''
gsfPFtracks.Fitter = 'GsfElectronFittingSmoother'
gsfPFtracks.Propagator = 'fwdElectronPropagator'
gsfPFtracks.src = 'gsfElCandidates'
gsfPFtracks.TTRHBuilder = 'WithTrackAngle'
gsfPFtracks.TrajectoryInEvent = True



# Electron propagators and estimators
# Looser chi2 estimator for electron trajectory building
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
electronEstimatorChi2 = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone()

# TrajectoryBuilder
import RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi
TrajectoryBuilderForElectronsinJets = RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi.CkfTrajectoryBuilder.clone()

TrajectoryBuilderForElectronsinJets.ComponentName = 'TrajectoryBuilderForElectronsinJets'
TrajectoryBuilderForElectronsinJets.trajectoryFilterName = 'TrajectoryFilterForPixelMatchGsfElectrons'
TrajectoryBuilderForElectronsinJets.maxCand = 3
TrajectoryBuilderForElectronsinJets.intermediateCleaning = False
TrajectoryBuilderForElectronsinJets.propagatorAlong = 'fwdGsfElectronPropagator'
TrajectoryBuilderForElectronsinJets.propagatorOpposite = 'bwdGsfElectronPropagator'
TrajectoryBuilderForElectronsinJets.estimator = 'electronEstimatorChi2'
TrajectoryBuilderForElectronsinJets.MeasurementTrackerName = ''
TrajectoryBuilderForElectronsinJets.lostHitPenalty = 100.
TrajectoryBuilderForElectronsinJets.alwaysUseInvalidHits = True
TrajectoryBuilderForElectronsinJets.TTRHBuilder = 'WithTrackAngle'
TrajectoryBuilderForElectronsinJets.updator = 'KFUpdator'
electronEstimatorChi2.ComponentName = 'electronEstimatorChi2'

electronEstimatorChi2.MaxChi2 = 2000.
electronEstimatorChi2.nSigma = 3.
