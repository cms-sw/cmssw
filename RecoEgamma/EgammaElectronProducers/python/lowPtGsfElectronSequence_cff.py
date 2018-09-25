import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to make low pT electrons.
# Several steps are cloned and modified to "open up" the sequence: 
# tracker-driven electron seeds, KF track candidates, GSF tracks.
#==============================================================================

# Tracker-driven seeds
from RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cfi import *
trackerDrivenElectronSeedsOpen = trackerDrivenElectronSeeds.clone()
trackerDrivenElectronSeedsOpen.PassThrough = True
trackerDrivenElectronSeedsOpen.PtThresholdSavePreId = 0.
trackerDrivenElectronSeedsOpen.MinPt = 0.5

# Above relies on the following default configurations !
# (Whatever module produces generalTracks ...)

# Electron (KF) track candidates
from TrackingTools.GsfTracking.CkfElectronCandidateMaker_cff import *
TrajectoryFilterForElectronsOpen = TrajectoryFilterForElectrons.clone()
TrajectoryFilterForElectronsOpen.minPt = 0.5
TrajectoryBuilderForElectronsOpen = TrajectoryBuilderForElectrons.clone()
TrajectoryBuilderForElectronsOpen.trajectoryFilter.refToPSet_ = "TrajectoryFilterForElectronsOpen"
electronCkfTrackCandidatesOpen = electronCkfTrackCandidates.clone()
electronCkfTrackCandidatesOpen.TrajectoryBuilderPSet.refToPSet_ = "TrajectoryBuilderForElectronsOpen"
electronCkfTrackCandidatesOpen.src = cms.InputTag("trackerDrivenElectronSeedsOpen","SeedsForGsf")

# GSF tracks
from TrackingTools.GsfTracking.GsfElectronGsfFit_cff import * 
electronGsfTracksOpen = electronGsfTracks.clone()
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
# Are the following necessary? 
#pfTrackElecOpen.PFEcalClusters  = "particleFlowClusterECAL"
pfTrackElecOpen.TrajInEvents = False
pfTrackElecOpen.ModeMomentum = True
pfTrackElecOpen.applyEGSelection = False
pfTrackElecOpen.applyGsfTrackCleaning = False
pfTrackElecOpen.applyAlsoGsfAngularCleaning = False
pfTrackElecOpen.useFifthStepForTrackerDrivenGsf = True
pfTrackElecOpen.useFifthStepForEcalDrivenGsf = True

# SuperCluster generator and matching to GSF tracks
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronSuperClusters_cfi import *
lowPtGsfElectronSuperClusters.gsfPfRecTracks = "pfTrackElecOpen"

# Above relies on the following default configurations !
# RecoParticleFlow/PFClusterProducer/python/particleFlowClusterECALUncorrected_cfi.py
# RecoParticleFlow/PFClusterProducer/python/particleFlowClusterECAL_cff.py
# (particleFlowClusterECAL_cfi is generated automatically)

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
