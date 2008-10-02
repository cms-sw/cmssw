import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.elecPreId_cff import *
from RecoParticleFlow.PFTracking.gsfSeedClean_cfi import *
from TrackingTools.GsfTracking.CkfElectronCandidates_cff import *
from TrackingTools.GsfTracking.GsfElectrons_cff import *
from RecoParticleFlow.PFTracking.pfNuclear_cfi import *
from RecoEgamma.EgammaElectronProducers.gsfElectronCkfTrackCandidateMaker_cff import *
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
gsfElCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import TrackingTools.GsfTracking.GsfElectronFit_cfi
gsfPFtracks = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *

from RecoParticleFlow.PFTracking.pfConversions_cfi import *

#TRAJECTORIES IN THE EVENT


#UNCOMMENT THE LINES THAT START WITH #DON# IN ORDER TO ADD CONVERSION FROM PF CLUSTERS
#DON#from RecoEgamma.EgammaPhotonProducers.softConversionSequence_cff import *
#DON#softConversionIOTracks.TrajectoryInEvent = cms.bool(True)
#DON#softConversionOITracks.TrajectoryInEvent = cms.bool(True)
#DON#pfConversions.OtherConversionCollection =cms.VInputTag(cms.InputTag("softConversions:softConversionCollection"))
#DON#pfConversions.OtherOutInCollection      =           cms.VInputTag(cms.InputTag("softConversionOITracks"))
#DON#pfConversions.OtherInOutCollection      =           cms.VInputTag(cms.InputTag("softConversionIOTracks"))

#UNCOMMENT THE LINES THAT START WITH #HON# IN ORDER TO ADD CONVERSION FROM PF CLUSTERS
#HON#from RecoEgamma.EgammaPhotonProducers.trackerOnlyConversionSequence_cff import *

#HON#pfConversions.OtherConversionCollection =cms.VInputTag(cms.InputTag("trackerOnlyConversions"))
#HON#pfConversions.OtherOutInCollection      =           cms.VInputTag(cms.InputTag("generalTracks"))
#HON#pfConversions.OtherInOutCollection      =           cms.VInputTag(cms.InputTag("generalTracks"))

particleFlowTrackWithConversion =cms.Sequence(
    elecPreId*
    gsfSeedclean*
    gsfElCandidates*
    gsfPFtracks*
    pfTrackElec*
    #HON#trackerOnlyConversionSequence*
    #DON#    softConversionSequence*
    pfConversions
    )


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
