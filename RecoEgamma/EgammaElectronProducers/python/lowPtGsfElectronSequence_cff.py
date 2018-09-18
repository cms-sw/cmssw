import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to make low pT electrons.
# Several steps are cloned and modified to "open up" the sequence: 
# tracker-driven electron seeds, KF track candidates, GSF tracks.
#==============================================================================

# Imports
from RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cfi import *
from TrackingTools.GsfTracking.CkfElectronCandidateMaker_cff import *
from TrackingTools.GsfTracking.GsfElectronGsfFit_cff import * 
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronCores_cfi import *
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectrons_cfi import *

# Tracker-driven seeds
trackerDrivenElectronSeedsOpen = trackerDrivenElectronSeeds.copy()
trackerDrivenElectronSeedsOpen.PassThrough = True
trackerDrivenElectronSeedsOpen.PtThresholdSavePreId = 0.
trackerDrivenElectronSeedsOpen.MinPt = 0.5

# Trajectories for KF track candidates
TrajectoryFilterForElectronsOpen = TrajectoryFilterForElectrons.copy()
TrajectoryFilterForElectronsOpen.minPt = 0.5

# Electron (KF) track candidates
TrajectoryFilterForElectronsOpen = TrajectoryFilterForElectrons.clone()
TrajectoryFilterForElectronsOpen.minPt = 0.5
TrajectoryBuilderForElectronsOpen = TrajectoryBuilderForElectrons.clone()
TrajectoryBuilderForElectronsOpen.trajectoryFilter.refToPSet_ = 'TrajectoryFilterForElectronsOpen'
electronCkfTrackCandidatesOpen = electronCkfTrackCandidates.copy()
electronCkfTrackCandidatesOpen.TrajectoryBuilderPSet.refToPSet_ = 'TrajectoryBuilderForElectronsOpen'
electronCkfTrackCandidatesOpen.src = cms.InputTag('trackerDrivenElectronSeedsOpen','SeedsForGsf')

# GSF tracks
electronGsfTracksOpen = electronGsfTracks.copy()
electronGsfTracksOpen.src = 'electronCkfTrackCandidatesOpen'

# Low pT electrons
lowPtGsfElectronCoresOpen = lowPtGsfElectronCores.copy()
lowPtGsfElectronCoresOpen.gsfTracks = "electronGsfTracksOpen"
lowPtGsfElectronsOpen = lowPtGsfElectrons.copy()
lowPtGsfElectronsOpen.gsfElectronCoresTag = "lowPtGsfElectronCoresOpen"

# Full Open sequence 
lowPtGsfElectronSequence = cms.Sequence(trackerDrivenElectronSeedsOpen+
                                        electronCkfTrackCandidatesOpen+
                                        electronGsfTracksOpen+
                                        lowPtGsfElectronCoresOpen+
                                        lowPtGsfElectronsOpen
                                        )
