import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to make low pT electrons.
# Several steps are cloned and modified to "open up" the sequence: 
# tracker-driven electron seeds, KF track candidates, GSF tracks.
#==============================================================================

# Tracker-driven seeds
from RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cfi import *
trackerDrivenElectronSeedsOpen = trackerDrivenElectronSeeds.copy()
trackerDrivenElectronSeedsOpen.PassThrough = True
trackerDrivenElectronSeedsOpen.PtThresholdSavePreId = 0.
trackerDrivenElectronSeedsOpen.MinPt = 0.5

# Electron (KF) track candidates
from TrackingTools.GsfTracking.CkfElectronCandidateMaker_cff import *
TrajectoryFilterForElectronsOpen = TrajectoryFilterForElectrons.clone()
TrajectoryFilterForElectronsOpen.minPt = 0.5
TrajectoryBuilderForElectronsOpen = TrajectoryBuilderForElectrons.clone()
TrajectoryBuilderForElectronsOpen.trajectoryFilter.refToPSet_ = "TrajectoryFilterForElectronsOpen"
electronCkfTrackCandidatesOpen = electronCkfTrackCandidates.copy()
electronCkfTrackCandidatesOpen.TrajectoryBuilderPSet.refToPSet_ = "TrajectoryBuilderForElectronsOpen"
electronCkfTrackCandidatesOpen.src = cms.InputTag("trackerDrivenElectronSeedsOpen","SeedsForGsf")

# GSF tracks
from TrackingTools.GsfTracking.GsfElectronGsfFit_cff import * 
electronGsfTracksOpen = electronGsfTracks.copy()
electronGsfTracksOpen.src = "electronCkfTrackCandidatesOpen"

# PFTracks
from RecoParticleFlow.PFTracking.pfTrack_cfi import *
pfTrackOpen = pfTrack.copy()
pfTrackOpen.GsfTrackModuleLabel = "electronGsfTracksOpen"

# PFGSFTracks
from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
pfTrackElecOpen = pfTrackElec.copy()
pfTrackElecOpen.applyGsfTrackCleaning = False
pfTrackElecOpen.GsfTrackModuleLabel = "electronGsfTracksOpen"
pfTrackElecOpen.PFRecTrackLabel = "pfTrackOpen"
#pfTrackElecOpen.PFEcalClusters  = "particleFlowClusterECAL"

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
                                        lowPtGsfElectronCores+
                                        lowPtGsfElectrons
                                        )
