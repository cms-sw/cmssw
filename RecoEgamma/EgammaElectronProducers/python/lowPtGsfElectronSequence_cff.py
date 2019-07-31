import FWCore.ParameterSet.Config as cms

# Modifier for FastSim
from Configuration.Eras.Modifier_fastSim_cff import fastSim

# PFRecTracks from generalTracks
from RecoParticleFlow.PFTracking.pfTrack_cfi import *
lowPtGsfElePfTracks = pfTrack.clone()
lowPtGsfElePfTracks.TkColList = ['generalTracks']
lowPtGsfElePfTracks.GsfTracksInEvents = False
lowPtGsfElePfTracks.GsfTrackModuleLabel = ''

fastSim.toModify(lowPtGsfElePfTracks,TkColList = ['generalTracksBeforeMixing'])

# Low pT ElectronSeeds
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronSeeds_cfi import *

# Electron track candidates
from TrackingTools.GsfTracking.CkfElectronCandidateMaker_cff import *
lowPtGsfEleTrajectoryFilter = TrajectoryFilterForElectrons.clone()
lowPtGsfEleTrajectoryFilter.minPt = 0.
lowPtGsfEleTrajectoryFilter.minimumNumberOfHits = 3
lowPtGsfEleTrajectoryBuilder = TrajectoryBuilderForElectrons.clone()
lowPtGsfEleTrajectoryBuilder.trajectoryFilter.refToPSet_ = 'lowPtGsfEleTrajectoryFilter'
lowPtGsfEleCkfTrackCandidates = electronCkfTrackCandidates.clone()
lowPtGsfEleCkfTrackCandidates.TrajectoryBuilderPSet.refToPSet_ = 'lowPtGsfEleTrajectoryBuilder'
lowPtGsfEleCkfTrackCandidates.src = 'lowPtGsfElectronSeeds'

import FastSimulation.Tracking.electronCkfTrackCandidates_cff
fastLowPtGsfTkfTrackCandidates = FastSimulation.Tracking.electronCkfTrackCandidates_cff.electronCkfTrackCandidates.clone(src = cms.InputTag("lowPtGsfElectronSeeds"))

# GsfTracks
from TrackingTools.GsfTracking.GsfElectronGsfFit_cff import *
lowPtGsfEleFittingSmoother = GsfElectronFittingSmoother.clone()
lowPtGsfEleFittingSmoother.ComponentName = 'lowPtGsfEleFittingSmoother'
lowPtGsfEleFittingSmoother.MinNumberOfHits = 2
from TrackingTools.GsfTracking.GsfElectronGsfFit_cff import * 
lowPtGsfEleGsfTracks = electronGsfTracks.clone()
lowPtGsfEleGsfTracks.Fitter = 'lowPtGsfEleFittingSmoother'
lowPtGsfEleGsfTracks.src = 'lowPtGsfEleCkfTrackCandidates'

fastSim.toModify(lowPtGsfEleGsfTracks,src = cms.InputTag("fastLowPtGsfTkfTrackCandidates"))

# GSFTrack to track association
from RecoEgamma.EgammaElectronProducers.lowPtGsfToTrackLinks_cfi import lowPtGsfToTrackLinks

# GsfPFRecTracks
from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
lowPtGsfElePfGsfTracks = pfTrackElec.clone()
lowPtGsfElePfGsfTracks.GsfTrackModuleLabel = 'lowPtGsfEleGsfTracks'
lowPtGsfElePfGsfTracks.PFRecTrackLabel = 'lowPtGsfElePfTracks'
lowPtGsfElePfGsfTracks.applyGsfTrackCleaning = False
lowPtGsfElePfGsfTracks.useFifthStepForTrackerDrivenGsf = True

# SuperCluster generator and matching to GSF tracks
# Below relies on the following default configurations:
# RecoParticleFlow/PFClusterProducer/python/particleFlowClusterECALUncorrected_cfi.py
# RecoParticleFlow/PFClusterProducer/python/particleFlowClusterECAL_cff.py
# (particleFlowClusterECAL_cfi is generated automatically)
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronSuperClusters_cfi import lowPtGsfElectronSuperClusters

# Low pT electron cores
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronCores_cff import lowPtGsfElectronCores

# Low pT electrons
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectrons_cfi import *

# Low pT Electron value maps
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronSeedValueMaps_cfi import lowPtGsfElectronSeedValueMaps

# Low pT Electron ID
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronID_cfi import lowPtGsfElectronID


# Full sequence 
lowPtGsfElectronTask = cms.Task(lowPtGsfElePfTracks,
                                lowPtGsfElectronSeeds,
                                lowPtGsfEleCkfTrackCandidates,
                                lowPtGsfEleGsfTracks,
                                lowPtGsfToTrackLinks,
                                lowPtGsfElePfGsfTracks,
                                lowPtGsfElectronSuperClusters,
                                lowPtGsfElectronCores,
                                lowPtGsfElectrons,
                                lowPtGsfElectronSeedValueMaps,
                                lowPtGsfElectronID
                                )
lowPtGsfElectronSequence = cms.Sequence(lowPtGsfElectronTask)

_fastSim_lowPtGsfElectronTask = lowPtGsfElectronTask.copy()
_fastSim_lowPtGsfElectronTask.replace(lowPtGsfElectronSeeds, cms.Task(lowPtGsfElectronSeedsTmp,lowPtGsfElectronSeeds))
_fastSim_lowPtGsfElectronTask.replace(lowPtGsfEleCkfTrackCandidates, fastLowPtGsfTkfTrackCandidates)
fastSim.toReplaceWith(lowPtGsfElectronTask, _fastSim_lowPtGsfElectronTask)
