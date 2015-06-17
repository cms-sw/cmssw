import FWCore.ParameterSet.Config as cms

#
# create a sequence with all required modules and sources needed to make
# pixel based electrons
#
# NB: it assumes that ECAL clusters (hybrid) are in the event
#
#
# modules to make seeds, tracks and electrons

# Cluster-seeded pixel pairs
#import FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi


#from FastSimulation.Configuration.blockHLT_1E31_cff import *

###new path from XXXX_49
#hltL1NonIsoLargeElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi.fastElectronSeeds.clone()
#hltL1NonIsoLargeElectronPixelSeeds.SeedConfiguration = cms.PSet(
#    block_hltL1NonIsoLargeElectronPixelSeeds
#)
#hltL1NonIsoLargeElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1NonIsolated'
#hltL1NonIsoLargeElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated'
#
#
#hltL1NonIsoLargeWindowElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi.fastElectronSeeds.clone()
#hltL1NonIsoLargeWindowElectronPixelSeeds.SeedConfiguration = cms.PSet(
#    block_hltL1NonIsoLargeWindowElectronPixelSeeds
#)
#hltL1NonIsoLargeWindowElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1NonIsolated'
#hltL1NonIsoLargeWindowElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated'

# (Not-so) Regional Tracking
from FastSimulation.Tracking.GlobalPixelTracking_cff import *

# Track candidate
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltCkfL1NonIsoLargeWindowTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltCkfL1NonIsoLargeWindowTrackCandidates.src = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds")
hltCkfL1NonIsoLargeWindowTrackCandidates.TrackProducers = cms.VInputTag(cms.InputTag("hltCtfL1NonIsoWithMaterialTracks"))
hltCkfL1NonIsoLargeWindowTrackCandidates.SplitHits = False


# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

ctfL1NonIsoLargeWindowTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
ctfL1NonIsoLargeWindowTracks.src = 'hltCkfL1NonIsoLargeWindowTrackCandidates'
ctfL1NonIsoLargeWindowTracks.TTRHBuilder = 'WithoutRefit'
ctfL1NonIsoLargeWindowTracks.Fitter = 'KFFittingSmootherForElectrons'
ctfL1NonIsoLargeWindowTracks.Propagator = 'PropagatorWithMaterial'

# Sequence
HLTPixelMatchLargeWindowElectronL1NonIsoTrackingSequence = cms.Sequence(hltCkfL1NonIsoLargeWindowTrackCandidates+
                                                                        ctfL1NonIsoLargeWindowTracks+
                                                                        cms.SequencePlaceholder("hltPixelMatchLargeWindowElectronsL1NonIso"))

hltL1NonIsoLargeWindowElectronPixelSeedsSequence = cms.Sequence(globalPixelTracking+
                                                                cms.SequencePlaceholder("hltL1NonIsoLargeWindowElectronPixelSeeds"))
