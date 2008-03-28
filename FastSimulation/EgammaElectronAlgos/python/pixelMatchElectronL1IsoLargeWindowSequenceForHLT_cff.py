import FWCore.ParameterSet.Config as cms

# $Id: pixelMatchElectronL1IsoLargeWindowSequenceForHLT.cff,v 1.6 2008/03/13 14:55:57 pjanot Exp $
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
l1IsoLargeWindowElectronPixelSeeds = copy.deepcopy(electronGSPixelSeeds)
from RecoEgamma.EgammaHLTProducers.pixelSeedConfigurationsForHLT_cfi import *
import copy
from FastSimulation.Tracking.TrackCandidateProducer_cfi import *
ckfL1IsoLargeWindowTrackCandidates = copy.deepcopy(trackCandidateProducer)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
# CTF track fit with material
ctfL1IsoLargeWindowTracks = copy.deepcopy(ctfWithMaterialTracks)
# Electron collection
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronsL1IsoLargeWindowForHLT_cff import *
ctfL1IsoLargeWindowWithMaterialTracks = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("ctfL1IsoLargeWindowTracks"), cms.InputTag("ctfL1IsoWithMaterialTracks"))
)

pixelMatchElectronL1IsoLargeWindowSequenceForHLT = cms.Sequence(l1IsoLargeWindowElectronPixelSeeds)
pixelMatchElectronL1IsoLargeWindowTrackingSequenceForHLT = cms.Sequence(ckfL1IsoLargeWindowTrackCandidates+ctfL1IsoLargeWindowTracks+ctfL1IsoLargeWindowWithMaterialTracks+pixelMatchElectronsL1IsoLargeWindowForHLT)
l1IsoLargeWindowElectronPixelSeeds.SeedConfiguration = cms.PSet(
    l1IsoLargeWindowElectronPixelSeedConfiguration
)
l1IsoLargeWindowElectronPixelSeeds.superClusterBarrel = 'correctedHybridSuperClustersL1Isolated'
l1IsoLargeWindowElectronPixelSeeds.superClusterEndcap = 'correctedEndcapSuperClustersWithPreshowerL1Isolated'
ckfL1IsoLargeWindowTrackCandidates.SeedProducer = cms.InputTag("l1IsoLargeWindowElectronPixelSeeds")
ckfL1IsoLargeWindowTrackCandidates.TrackProducer = cms.InputTag("ctfL1IsoWithMaterialTracks")
ckfL1IsoLargeWindowTrackCandidates.MaxNumberOfCrossedLayers = 999
ckfL1IsoLargeWindowTrackCandidates.SeedCleaning = True
ctfL1IsoLargeWindowTracks.src = 'ckfL1IsoLargeWindowTrackCandidates'
ctfL1IsoLargeWindowTracks.TTRHBuilder = 'WithoutRefit'
ctfL1IsoLargeWindowTracks.Fitter = 'KFFittingSmoother'
ctfL1IsoLargeWindowTracks.Propagator = 'PropagatorWithMaterial'

