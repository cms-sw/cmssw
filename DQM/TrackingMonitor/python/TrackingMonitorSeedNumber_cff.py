import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
from DQM.TrackingMonitor.TrackingMonitorSeed_cfi import *

TrackMonStep0 = TrackMonSeed.clone(
    TrackProducer = "generalTracks",
    SeedProducer = "initialStepSeeds",
    TCProducer = "initialStepTrackCandidates",
    AlgoName = 'initialStep',
    TkSeedSizeBin = 100, # could be 50 ?
    TkSeedSizeMax = 5000.,
    TkSeedSizeMin = 0.,
    NClusPxBin = 100,
    NClusPxMax = 20000.,
    ClusterLabels = ('Pix',)
)

TrackMonStep1 = TrackMonSeed.clone(
    TrackProducer = "generalTracks",
    SeedProducer = "lowPtTripletStepSeeds",
    TCProducer = "lowPtTripletStepTrackCandidates",
    AlgoName = 'lowPtTripletStep',
    TkSeedSizeBin = 100,
    TkSeedSizeMax = 30000.,                         
    TkSeedSizeMin = 0.,
    NClusPxBin = 100,
    NClusPxMax = 20000.,
    ClusterLabels = ('Pix',)
)

TrackMonStep2 = TrackMonSeed.clone(
    TrackProducer = "generalTracks",
    SeedProducer = "pixelPairStepSeeds",
    TCProducer = "pixelPairStepTrackCandidates",
    AlgoName = 'pixelPairStep',
    TkSeedSizeBin = 400,
    TkSeedSizeMax = 100000.,                         
    TkSeedSizeMin = 0.,
    TCSizeMax = 199.5,
    NClusPxBin = 100,
    NClusPxMax = 20000.,
    ClusterLabels = ('Pix',)
)

TrackMonStep3 = TrackMonSeed.clone(
    TrackProducer = "generalTracks",
    SeedProducer = "detachedTripletStepSeeds",
    TCProducer = "detachedTripletStepTrackCandidates",
    AlgoName = 'detachedTripletStep',
    TkSeedSizeBin = 100,
    TkSeedSizeMax = 30000.,                         
    TkSeedSizeMin = 0.,
    NClusPxBin = 100,
    NClusPxMax = 20000.,
    ClusterLabels = ('Pix',)
)

TrackMonStep4 = TrackMonSeed.clone(
    TrackProducer = "generalTracks",
    SeedProducer = "mixedTripletStepSeeds",
    TCProducer = "mixedTripletStepTrackCandidates",
    AlgoName = 'mixedTripletStep',
    TkSeedSizeBin = 400,
    TkSeedSizeMax = 200000.,                         
    TkSeedSizeMin = 0.,
    TCSizeMax = 199.5,
    NClusStrBin = 500,
    NClusStrMax = 100000.,
    ClusterLabels = ('Tot',)
)

TrackMonStep5 = TrackMonSeed.clone(
    TrackProducer = "generalTracks",
    SeedProducer = "pixelLessStepSeeds",
    TCProducer = "pixelLessStepTrackCandidates",
    AlgoName = 'pixelLessStep',
    TkSeedSizeBin = 400,
    TkSeedSizeMax = 200000.,
    TkSeedSizeMin = 0.,
    NClusStrBin = 500,
    NClusStrMax = 100000.,
    ClusterLabels = ('Strip',)
)

TrackMonStep6 = TrackMonSeed.clone(
    TrackProducer = "generalTracks",
    SeedProducer = "tobTecStepSeeds",
    TCProducer = "tobTecStepTrackCandidates",
    AlgoName = 'tobTecStep',
    TkSeedSizeBin = 400,
    TkSeedSizeMax = 100000.,                         
    TkSeedSizeMin = 0.,
    TCSizeMax = 199.5,
    NClusStrBin = 500,
    NClusStrMax = 100000.,
    ClusterLabels = ('Strip',)
)

TrackMonStep9 = TrackMonSeed.clone(
    TrackProducer = "generalTracks",
    SeedProducer = "muonSeededSeedsInOut",
    TCProducer = "muonSeededTrackCandidatesInOut",
    AlgoName = 'muonSeededStepInOut',
    TkSeedSizeBin = 15,
    TkSeedSizeMax = 14.5,                         
    TkSeedSizeMin = -0.5,
    TCSizeMax = 199.5,
    NClusStrBin = 500,
    NClusStrMax = 100000.,
    ClusterLabels = ('Strip',)
)

TrackMonStep10 = TrackMonSeed.clone(
    TrackProducer = "generalTracks",
    SeedProducer = "muonSeededSeedsOutIn",
    TCProducer = "muonSeededTrackCandidatesOutIn",
    AlgoName = 'muonSeededStepOutIn',
    TkSeedSizeBin = 15,
    TkSeedSizeMax = 14.5,                         
    TkSeedSizeMin = -0.5,
    TCSizeMax = 199.5,
    NClusStrBin = 500,
    NClusStrMax = 100000.,
    ClusterLabels = ('Strip',)
)

# out of the box
trackMonIterativeTracking2012 = cms.Sequence(
     TrackMonStep0
    * TrackMonStep1
    * TrackMonStep2
    * TrackMonStep3
    * TrackMonStep4
    * TrackMonStep5
    * TrackMonStep6
    * TrackMonStep9
    * TrackMonStep10
)



# all paths
trkmon = cms.Sequence(
      trackMonIterativeTracking2012
)

