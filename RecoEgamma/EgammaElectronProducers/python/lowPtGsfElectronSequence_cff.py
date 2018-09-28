import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to make low pT electrons.
# Several steps are cloned and modified to "open up" the sequence: 
# tracker-driven electron seeds, KF track candidates, GSF tracks.
#==============================================================================

# Tracker-driven seeds
# Below relies on default configuration for generalTracks
from RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cfi import *
trackerDrivenElectronSeedsOpen = trackerDrivenElectronSeeds.clone()
trackerDrivenElectronSeedsOpen.PassThrough = True
trackerDrivenElectronSeedsOpen.PtThresholdSavePreId = 0.
trackerDrivenElectronSeedsOpen.MinPt = 0.

# Electron (KF) track candidates
from TrackingTools.GsfTracking.CkfElectronCandidateMaker_cff import *
TrajectoryFilterForElectronsOpen = TrajectoryFilterForElectrons.clone()
TrajectoryFilterForElectronsOpen.minPt = 0.
TrajectoryFilterForElectronsOpen.minimumNumberOfHits = 3
TrajectoryBuilderForElectronsOpen = TrajectoryBuilderForElectrons.clone()
TrajectoryBuilderForElectronsOpen.trajectoryFilter.refToPSet_ = "TrajectoryFilterForElectronsOpen"
electronCkfTrackCandidatesOpen = electronCkfTrackCandidates.clone()
electronCkfTrackCandidatesOpen.TrajectoryBuilderPSet.refToPSet_ = "TrajectoryBuilderForElectronsOpen"
electronCkfTrackCandidatesOpen.src = "trackerDrivenElectronSeedsOpen:SeedsForGsf"


# GSF tracks
#from TrackingTools.GsfTracking.GsfElectronGsfFit_cff import *
#GsfElectronFittingSmootherOpen = GsfElectronFittingSmoother.clone()
#GsfElectronFittingSmootherOpen.ComponentName = 'GsfElectronFittingSmootherOpen'
#GsfElectronFittingSmootherOpen.MinNumberOfHits = 2
from TrackingTools.GsfTracking.GsfElectronGsfFit_cff import * 
electronGsfTracksOpen = electronGsfTracks.clone()
#electronGsfTracks.Fitter = 'GsfElectronFittingSmoother'
electronGsfTracksOpen.src = "electronCkfTrackCandidatesOpen"

# PFTracks
from RecoParticleFlow.PFTracking.pfTrack_cfi import *
pfTrackOpen = pfTrack.clone()
pfTrackOpen.GsfTrackModuleLabel = "electronGsfTracksOpen"

# PFGSFTracks
from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
pfTrackElecOpen = pfTrackElec.clone()
pfTrackElecOpen.applyGsfTrackCleaning = False
pfTrackElecOpen.GsfTrackModuleLabel = "electronGsfTracksOpen"
pfTrackElecOpen.PFRecTrackLabel = "pfTrackOpen"
# Are ALL of the following necessary? 
pfTrackElecOpen.TrajInEvents = False
pfTrackElecOpen.ModeMomentum = True
pfTrackElecOpen.applyEGSelection = False
pfTrackElecOpen.applyGsfTrackCleaning = False
pfTrackElecOpen.applyAlsoGsfAngularCleaning = False
pfTrackElecOpen.useFifthStepForTrackerDrivenGsf = True
pfTrackElecOpen.useFifthStepForEcalDrivenGsf = True

# SuperCluster generator and matching to GSF tracks
# Below relies on the following default configurations:
# RecoParticleFlow/PFClusterProducer/python/particleFlowClusterECALUncorrected_cfi.py
# RecoParticleFlow/PFClusterProducer/python/particleFlowClusterECAL_cff.py
# (particleFlowClusterECAL_cfi is generated automatically)
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronSuperClusters_cfi import *
lowPtGsfElectronSuperClusters.gsfPfRecTracks = "pfTrackElecOpen"

# Low pT electron cores
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronCores_cfi import *
lowPtGsfElectronCores.gsfPfRecTracks = "pfTrackElecOpen"
lowPtGsfElectronCores.gsfTracks = "electronGsfTracksOpen"
lowPtGsfElectronCores.useGsfPfRecTracks = True

# Low pT electrons
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectrons_cfi import *
lowPtGsfElectrons.gsfElectronCoresTag = "lowPtGsfElectronCores"
lowPtGsfElectrons.seedsTag = "trackerDrivenElectronSeedsOpen:SeedsForGsf"
lowPtGsfElectrons.gsfPfRecTracksTag = "pfTrackElecOpen"
# Ignore below for now
#lowPtGsfElectrons.ctfTracksCheck = False
#lowPtGsfElectrons.gedElectronMode= True
#lowPtGsfElectrons.PreSelectMVA = -1.e6
#lowPtGsfElectrons.MaxElePtForOnlyMVA = 1.e6
#lowPtGsfElectrons.ecalDrivenEcalEnergyFromClassBasedParameterization = False
#lowPtGsfElectrons.ecalDrivenEcalErrorFromClassBasedParameterization = False
#lowPtGsfElectrons.pureTrackerDrivenEcalErrorFromSimpleParameterization = False
#lowPtGsfElectrons.minMVA = -1.e6
#lowPtGsfElectrons.minMvaByPassForIsolated = -0.4
#lowPtGsfElectrons.minMVAPflow = -1.e6
#lowPtGsfElectrons.minMvaByPassForIsolatedPflow = -0.4

# Full Open sequence 
lowPtGsfElectronSequence = cms.Sequence(trackerDrivenElectronSeedsOpen+
                                        electronCkfTrackCandidatesOpen+
                                        electronGsfTracksOpen+
                                        pfTrackOpen+
                                        pfTrackElecOpen+
                                        lowPtGsfElectronSuperClusters+
                                        lowPtGsfElectronCores+
                                        lowPtGsfElectrons
                                        )
