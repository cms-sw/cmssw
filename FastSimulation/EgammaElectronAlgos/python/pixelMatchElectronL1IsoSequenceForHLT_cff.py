import FWCore.ParameterSet.Config as cms

# $Id: pixelMatchElectronL1IsoSequenceForHLT.cff,v 1.6 2008/03/13 14:55:57 pjanot Exp $
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
l1IsoElectronPixelSeeds = copy.deepcopy(electronGSPixelSeeds)
from RecoEgamma.EgammaHLTProducers.pixelSeedConfigurationsForHLT_cfi import *
import copy
from FastSimulation.Tracking.TrackCandidateProducer_cfi import *
# CKFTrackCandidateMaker
ckfL1IsoTrackCandidates = copy.deepcopy(trackCandidateProducer)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
# CTF track fit with material
ctfL1IsoWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
# Electron collection
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronsL1IsoForHLT_cff import *
pixelMatchElectronL1IsoSequenceForHLT = cms.Sequence(l1IsoElectronPixelSeeds)
l1IsoElectronPixelSeeds.SeedConfiguration = cms.PSet(
    l1IsoElectronPixelSeedConfiguration
)
l1IsoElectronPixelSeeds.superClusterBarrel = 'correctedHybridSuperClustersL1Isolated'
l1IsoElectronPixelSeeds.superClusterEndcap = 'correctedEndcapSuperClustersWithPreshowerL1Isolated'
ckfL1IsoTrackCandidates.SeedProducer = cms.InputTag("l1IsoElectronPixelSeeds")
ckfL1IsoTrackCandidates.TrackProducer = cms.InputTag("None","None")
ckfL1IsoTrackCandidates.MaxNumberOfCrossedLayers = 999
ckfL1IsoTrackCandidates.SeedCleaning = True
ctfL1IsoWithMaterialTracks.src = 'ckfL1IsoTrackCandidates'
ctfL1IsoWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
ctfL1IsoWithMaterialTracks.Fitter = 'KFFittingSmoother'
ctfL1IsoWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

