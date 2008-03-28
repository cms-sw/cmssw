import FWCore.ParameterSet.Config as cms

# $Id: pixelMatchElectronL1NonIsoSequenceForHLT.cff,v 1.6 2008/03/13 14:55:57 pjanot Exp $
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
l1NonIsoElectronPixelSeeds = copy.deepcopy(electronGSPixelSeeds)
from RecoEgamma.EgammaHLTProducers.pixelSeedConfigurationsForHLT_cfi import *
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronsL1NonIsoForHLT_cff import *
import copy
from FastSimulation.Tracking.TrackCandidateProducer_cfi import *
# CKFTrackCandidateMaker
ckfL1NonIsoTrackCandidates = copy.deepcopy(trackCandidateProducer)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
# CTF track fit with material
ctfL1NonIsoWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
pixelMatchElectronL1NonIsoSequenceForHLT = cms.Sequence(l1NonIsoElectronPixelSeeds)
l1NonIsoElectronPixelSeeds.SeedConfiguration = cms.PSet(
    l1NonIsoElectronPixelSeedConfiguration
)
l1NonIsoElectronPixelSeeds.superClusterBarrel = 'correctedHybridSuperClustersL1NonIsolated'
l1NonIsoElectronPixelSeeds.superClusterEndcap = 'correctedEndcapSuperClustersWithPreshowerL1NonIsolated'
ckfL1NonIsoTrackCandidates.SeedProducer = cms.InputTag("l1NonIsoElectronPixelSeeds")
ckfL1NonIsoTrackCandidates.TrackProducer = cms.InputTag("None","None")
ckfL1NonIsoTrackCandidates.MaxNumberOfCrossedLayers = 999
ckfL1NonIsoTrackCandidates.SeedCleaning = True
ctfL1NonIsoWithMaterialTracks.src = 'ckfL1NonIsoTrackCandidates'
ctfL1NonIsoWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
ctfL1NonIsoWithMaterialTracks.Fitter = 'KFFittingSmoother'
ctfL1NonIsoWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

