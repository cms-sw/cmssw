import FWCore.ParameterSet.Config as cms

# $Id: pixelMatchElectronL1NonIsoLargeWindowSequenceForHLT.cff,v 1.6 2008/03/13 14:55:57 pjanot Exp $
# create a sequence with all required modules and sources needed to make
# pixel based electrons
#
# NB: it assumes that ECAL clusters (hybrid) are in the event
#
#
# modules to make seeds, tracks and electrons
from RecoEgamma.EgammaHLTProducers.egammaHLTChi2MeasurementEstimatorESProducer_cff import *
import copy
from FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi import *
# Cluster-seeded pixel pairs
l1NonIsoLargeWindowElectronPixelSeeds = copy.deepcopy(electronGSPixelSeeds)
from RecoEgamma.EgammaHLTProducers.pixelSeedConfigurationsForHLT_cfi import *
import copy
from FastSimulation.Tracking.TrackCandidateProducer_cfi import *
ckfL1NonIsoLargeWindowTrackCandidates = copy.deepcopy(trackCandidateProducer)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
# CTF track fit with material
ctfL1NonIsoLargeWindowTracks = copy.deepcopy(ctfWithMaterialTracks)
# Electron collection
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronsL1NonIsoLargeWindowForHLT_cff import *
ctfL1NonIsoLargeWindowWithMaterialTracks = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("ctfL1NonIsoLargeWindowTracks"), cms.InputTag("ctfL1NonIsoWithMaterialTracks"))
)

pixelMatchElectronL1NonIsoLargeWindowSequenceForHLT = cms.Sequence(l1NonIsoLargeWindowElectronPixelSeeds)
pixelMatchElectronL1NonIsoLargeWindowTrackingSequenceForHLT = cms.Sequence(ckfL1NonIsoLargeWindowTrackCandidates+ctfL1NonIsoLargeWindowTracks+ctfL1NonIsoLargeWindowWithMaterialTracks+pixelMatchElectronsL1NonIsoLargeWindowForHLT)
l1NonIsoLargeWindowElectronPixelSeeds.SeedConfiguration = cms.PSet(
    l1NonIsoLargeWindowElectronPixelSeedConfiguration
)
l1NonIsoLargeWindowElectronPixelSeeds.superClusterBarrel = 'correctedHybridSuperClustersL1NonIsolated'
l1NonIsoLargeWindowElectronPixelSeeds.superClusterEndcap = 'correctedEndcapSuperClustersWithPreshowerL1NonIsolated'
ckfL1NonIsoLargeWindowTrackCandidates.SeedProducer = cms.InputTag("l1NonIsoLargeWindowElectronPixelSeeds")
ckfL1NonIsoLargeWindowTrackCandidates.TrackProducer = cms.InputTag("ctfL1NonIsoWithMaterialTracks")
ckfL1NonIsoLargeWindowTrackCandidates.MaxNumberOfCrossedLayers = 999
ckfL1NonIsoLargeWindowTrackCandidates.SeedCleaning = True
ctfL1NonIsoLargeWindowTracks.src = 'ckfL1NonIsoLargeWindowTrackCandidates'
ctfL1NonIsoLargeWindowTracks.TTRHBuilder = 'WithoutRefit'
ctfL1NonIsoLargeWindowTracks.Fitter = 'KFFittingSmoother'
ctfL1NonIsoLargeWindowTracks.Propagator = 'PropagatorWithMaterial'

