import FWCore.ParameterSet.Config as cms


#
# create a sequence with all required modules and sources needed to make
# pixel based electrons
#
# NB: it assumes that ECAL clusters (hybrid) are in the event
#
#
# modules to make seeds, tracks and electrons
# include "RecoEgamma/EgammaHLTProducers/data/egammaHLTChi2MeasurementEstimatorESProducer.cff"
# Cluster-seeded pixel pairs
#import FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi

# (Not-so) Regional Tracking
from FastSimulation.Tracking.GlobalPixelTracking_cff import *

#####put the hack! choose 1E31 that is more inclusive! 
#from FastSimulation.Configuration.blockHLT_1E31_cff import *
#####

##new path in XXXX_49 for 1E31 only
#hltL1IsoLargeElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi.fastElectronSeeds.clone()
#hltL1IsoLargeElectronPixelSeeds.SeedConfiguration = cms.PSet(
#    block_hltL1IsoLargeElectronPixelSeeds
#)
#hltL1IsoLargeElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1Isolated'
#hltL1IsoLargeElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated'
#
#
#
#hltL1IsoLargeWindowElectronPixelSeeds = FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi.fastElectronSeeds.clone()
#hltL1IsoLargeWindowElectronPixelSeeds.SeedConfiguration = cms.PSet(
#    block_hltL1IsoLargeWindowElectronPixelSeeds
#)
#hltL1IsoLargeWindowElectronPixelSeeds.barrelSuperClusters = 'hltCorrectedHybridSuperClustersL1Isolated'
#hltL1IsoLargeWindowElectronPixelSeeds.endcapSuperClusters = 'hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated'

# Track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
hltCkfL1IsoLargeWindowTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltCkfL1IsoLargeWindowTrackCandidates.src = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds")
hltCkfL1IsoLargeWindowTrackCandidates.TrackProducers = cms.VInputTag(cms.InputTag("hltCtfL1IsoWithMaterialTracks"))
hltCkfL1IsoLargeWindowTrackCandidates.SplitHits = False


# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
ctfL1IsoLargeWindowTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
ctfL1IsoLargeWindowTracks.src = 'hltCkfL1IsoLargeWindowTrackCandidates'
ctfL1IsoLargeWindowTracks.TTRHBuilder = 'WithoutRefit'
ctfL1IsoLargeWindowTracks.Fitter = 'KFFittingSmootherForElectrons'
ctfL1IsoLargeWindowTracks.Propagator = 'PropagatorWithMaterial'




HLTPixelMatchLargeWindowElectronL1IsoTrackingSequence = cms.Sequence(hltCkfL1IsoLargeWindowTrackCandidates+
                                                                     ctfL1IsoLargeWindowTracks+
                                                                     cms.SequencePlaceholder("hltPixelMatchLargeWindowElectronsL1Iso"))


hltL1IsoLargeWindowElectronPixelSeedsSequence = cms.Sequence(globalPixelTracking+
                                                             cms.SequencePlaceholder("hltL1IsoLargeWindowElectronPixelSeeds"))
