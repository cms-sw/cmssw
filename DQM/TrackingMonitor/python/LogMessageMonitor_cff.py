import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.LogMessageMonitor_cfi import *

LocalRecoLogMessageMon = LogMessageMon.clone(
    pluginsMonName = 'LocalReco',
    modules = ('siPixelDigis', 'siStripDigis', 'siPixelClusters', 'siStripClusters',), # siPixelDigis : SiPixelRawToDigi, siStripDigis : SiStripRawToDigi (SiStripRawToDigiUnpacker), siPixelClusters : SiPixelClusterProducer, siStripClusters : SiStripClusterizer
    categories = ('SiPixelRawToDigi', 'TooManyErrors', 'TooManyClusters',)
)

# apparentely there are not LogError in RecoLocalTracker/SubCollectionProducers/src/TrackClusterRemover.cc
ClusterizerLogMessageMon = LogMessageMon.clone(
    pluginsMonName = 'TrackClusterRemover',
    modules = ('detachedTripletStepClusters', 'lowPtTripletStepClusters', 'pixelPairStepClusters', 'mixedTripletStepClusters', 'pixelLessStepClusters', 'tobTecStepClusters',), # TrackClusterRemover
    categories = ()
)

# initialStepSeeds,lowPtTripletStepSeeds, pixelPairStepSeeds, detachedTripletStepSeeds, : TooManyClusters (SeedGeneratorFromRegionHitsEDProducer),
# photonConvTrajSeedFromSingleLeg : (PhotonConversionTrajectorySeedProducerFromSingleLeg)
SeedingLogMessageMon = LogMessageMon.clone(
    pluginsMonName = 'Seeding',
    modules = ('initialStepSeedsPreSplitting', 'initialStepSeeds', 'detachedTripletStepSeeds', 'lowPtTripletStepSeeds', 'pixelPairStepSeeds', 'mixedTripletStepSeedsA', 'mixedTripletStepSeedsB', 'pixelLessStepSeeds', 'tobTecStepSeeds', 'jetCoreRegionalStepSeeds', 'muonSeededSeedsOutIn', 'muonSeededSeedsInOut', 'photonConvTrajSeedFromSingleLeg',),
    categories = ('TooManyClusters', 'TooManyPairs', 'TooManyTriplets', 'TooManySeeds',)
)

# RecoTracker/CkfPattern/src/CkfTrackCandidateMakerBase.cc
TrackCandidateLogMessageMon = LogMessageMon.clone(
    pluginsMonName = 'TrackCandidate',
    modules = ('initialStepTrackCandidatesPreSplitting', 'initialStepTrackCandidates', 'detachedTripletStepTrackCandidates', 'lowPtTripletStepTrackCandidates', 'pixelPairStepTrackCandidates', 'mixedTripletStepTrackCandidates', 'pixelLessStepTrackCandidates', 'tobTecStepTrackCandidates', 'jetCoreRegionalStepTrackCandidates', 'muonSeededTrackCandidatesInOut', 'muonSeededTrackCandidatesOutIn', 'convTrackCandidates',),
    categories = ('TooManySeeds',)
)

# TrackProducer:FailedPropagation 
TrackFinderLogMessageMon = LogMessageMon.clone(
    pluginsMonName = 'TrackFinder',
    modules = ('pixelTracks', 'initialStepTracks', 'lowPtTripletStepTracks', 'pixelPairStepTracks', 'detachedTripletStepTracks', 'mixedTripletStepTracks', 'pixelLessStepTracks', 'tobTecStepTracks', 'jetCoreRegionalStepTracks', 'muonSeededTracksOutIn', 'muonSeededTracksInOut', 'convStepTracks', 'generalTracks',),
    categories = ('FailedPropagation', 'RKPropagatorInS',)
)

FullIterTrackingLogMessageMon = LogMessageMon.clone(
    pluginsMonName = 'FullIterTracking',
    modules = (
        'initialStepSeeds_iter0',
        'initialStepTrackCandidates_iter0',
        'initialStepTracks_iter0',
        'lowPtTripletStepSeeds_iter1',
        'lowPtTripletStepTrackCandidates_iter1',
        'lowPtTripletStepTracks_iter1',
        'pixelPairStepSeeds_iter2',
        'pixelPairStepTrackCandidates_iter2',
        'pixelPairStepTracks_iter2',
        'detachedTripletStepSeeds_iter3',
        'detachedTripletStepTrackCandidates_iter3',
        'detachedTripletStepTracks_iter3',
        'mixedTripletStepSeedsA_iter4',
        'mixedTripletStepSeedsB_iter4',
        'mixedTripletStepTrackCandidates_iter4',
        'mixedTripletStepTracks_iter4',
        'pixelLessStepSeeds_iter5',
        'pixelLessStepTrackCandidates_iter5',
        'pixelLessStepTracks_iter5',
        'tobTecStepSeeds_iter6',
        'tobTecStepTrackCandidates_iter6',
        'tobTecStepTracks_iter6',
        'photonConvTrajSeedFromSingleLeg',
        'convTrackCandidates',
        'convStepTracks',
    ),
    categories = ('TooManyClusters', 'TooManyPairs', 'TooManyTriplets', 'TooManySeeds',)
)

IterTrackingLogMessageMon = LogMessageMon.clone(
    pluginsMonName = 'IterTracking',
    modules = (
        'initialStepSeeds_iter0',
        'initialStepTrackCandidates_iter0',
        'initialStepTracks_iter0',
        'lowPtTripletStepSeeds_iter1',
        'lowPtTripletStepTrackCandidates_iter1',
        'lowPtTripletStepTracks_iter1',
        'pixelPairStepSeeds_iter2',
        'pixelPairStepTrackCandidates_iter2',
        'pixelPairStepTracks_iter2',
        'detachedTripletStepSeeds_iter3',
        'detachedTripletStepTrackCandidates_iter3',
        'detachedTripletStepTracks_iter3',
        'mixedTripletStepSeedsA_iter4',
        'mixedTripletStepSeedsB_iter4',
        'mixedTripletStepTrackCandidates_iter4',
        'mixedTripletStepTracks_iter4',
        'pixelLessStepSeeds_iter5',
        'pixelLessStepTrackCandidates_iter5',
        'pixelLessStepTracks_iter5',
        'tobTecStepSeeds_iter6',
        'tobTecStepTrackCandidates_iter6',
        'tobTecStepTracks_iter6',
    ),
    categories = ('TooManyClusters', 'TooManyPairs', 'TooManyTriplets', 'TooManySeeds',)
)

ConversionLogMessageMon = LogMessageMon.clone(
    pluginsMonName = 'Conversion',
    modules = ('photonConvTrajSeedFromSingleLeg', 'convTrackCandidates', 'convStepTracks',),
    categories = ('TooManyClusters', 'TooManyPairs', 'TooManyTriplets', 'TooManySeeds',)
)

