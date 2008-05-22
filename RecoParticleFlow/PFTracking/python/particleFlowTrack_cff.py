import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.elecPreId_cff import *
from TrackingTools.GsfTracking.CkfElectronCandidates_cff import *
from TrackingTools.GsfTracking.GsfElectrons_cff import *
from RecoParticleFlow.PFTracking.pfNuclear_cfi import *
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
gsfElCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import TrackingTools.GsfTracking.GsfElectronFit_cfi
gsfPFtracks = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
particleFlowTrack = cms.Sequence(elecPreId*gsfElCandidates*gsfPFtracks*pfTrackElec)
particleFlowTrackWithNuclear = cms.Sequence(elecPreId*gsfElCandidates*gsfPFtracks*pfTrackElec*pfNuclear)
gsfElCandidates.TrajectoryBuilder = 'CkfElectronTrajectoryBuilder'
gsfElCandidates.SeedProducer = 'elecpreid'
gsfElCandidates.SeedLabel = 'SeedsForGsf'
gsfPFtracks.Fitter = 'GsfElectronFittingSmoother'
gsfPFtracks.Propagator = 'fwdElectronPropagator'
gsfPFtracks.src = 'gsfElCandidates'
gsfPFtracks.TTRHBuilder = 'WithTrackAngle'
gsfPFtracks.TrajectoryInEvent = True

