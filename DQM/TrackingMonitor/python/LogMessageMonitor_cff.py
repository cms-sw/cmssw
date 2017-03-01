import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.LogMessageMonitor_cfi

LocalRecoLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
LocalRecoLogMessageMon.pluginsMonName = cms.string ( 'LocalReco' )
LocalRecoLogMessageMon.modules        = cms.vstring( 'siPixelDigis', 'siStripDigis', 'siPixelClusters', 'siStripClusters' ) # siPixelDigis : SiPixelRawToDigi, siStripDigis : SiStripRawToDigi (SiStripRawToDigiUnpacker), siPixelClusters : SiPixelClusterProducer, siStripClusters : SiStripClusterizer
LocalRecoLogMessageMon.categories     = cms.vstring( 'SiPixelRawToDigi', 'TooManyErrors', 'TooManyClusters' )

# apparentely there are not LogError in RecoLocalTracker/SubCollectionProducers/src/TrackClusterRemover.cc
ClusterizerLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
ClusterizerLogMessageMon.pluginsMonName = cms.string ( 'TrackClusterRemover' )
ClusterizerLogMessageMon.modules        = cms.vstring( 'detachedTripletStepClusters', 'lowPtTripletStepClusters', 'pixelPairStepClusters', 'mixedTripletStepClusters', 'pixelLessStepClusters', 'tobTecStepClusters' ) # TrackClusterRemover
ClusterizerLogMessageMon.categories     = cms.vstring(  )

# initialStepSeeds,lowPtTripletStepSeeds, pixelPairStepSeeds, detachedTripletStepSeeds, : TooManyClusters (SeedGeneratorFromRegionHitsEDProducer),
# photonConvTrajSeedFromSingleLeg : (PhotonConversionTrajectorySeedProducerFromSingleLeg)
SeedingLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
SeedingLogMessageMon.pluginsMonName = cms.string ( 'Seeding' ) 
SeedingLogMessageMon.modules        = cms.vstring( 'initialStepSeedsPreSplitting', 'initialStepSeeds', 'detachedTripletStepSeeds', 'lowPtTripletStepSeeds', 'pixelPairStepSeeds', 'mixedTripletStepSeedsA', 'mixedTripletStepSeedsB', 'pixelLessStepSeeds', 'tobTecStepSeeds', 'jetCoreRegionalStepSeeds', 'muonSeededSeedsOutIn', 'muonSeededSeedsInOut', 'photonConvTrajSeedFromSingleLeg')
SeedingLogMessageMon.categories     = cms.vstring( 'TooManyClusters', 'TooManyPairs', 'TooManyTriplets', 'TooManySeeds' )

# RecoTracker/CkfPattern/src/CkfTrackCandidateMakerBase.cc
TrackCandidateLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
TrackCandidateLogMessageMon.pluginsMonName = cms.string ( 'TrackCandidate' ) 
TrackCandidateLogMessageMon.modules        = cms.vstring( 'initialStepTrackCandidatesPreSplitting', 'initialStepTrackCandidates', 'detachedTripletStepTrackCandidates', 'lowPtTripletStepTrackCandidates', 'pixelPairStepTrackCandidates', 'mixedTripletStepTrackCandidates', 'pixelLessStepTrackCandidates', 'tobTecStepTrackCandidates', 'jetCoreRegionalStepTrackCandidates', 'muonSeededTrackCandidatesInOut', 'muonSeededTrackCandidatesOutIn', 'convTrackCandidates' )
TrackCandidateLogMessageMon.categories     = cms.vstring( 'TooManySeeds' )

# TrackProducer:FailedPropagation 
TrackFinderLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
TrackFinderLogMessageMon.pluginsMonName = cms.string ( 'TrackFinder' ) 
TrackFinderLogMessageMon.modules        = cms.vstring( 'pixelTracks', 'initialStepTracks', 'lowPtTripletStepTracks', 'pixelPairStepTracks', 'detachedTripletStepTracks', 'mixedTripletStepTracks', 'pixelLessStepTracks', 'tobTecStepTracks', 'jetCoreRegionalStepTracks', 'muonSeededTracksOutIn', 'muonSeededTracksInOut', 'convStepTracks', 'generalTracks' )
TrackFinderLogMessageMon.categories     = cms.vstring(
    'FailedPropagation', 'RKPropagatorInS'
)

FullIterTrackingLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
FullIterTrackingLogMessageMon.pluginsMonName = cms.string ( 'FullIterTracking' ) 
FullIterTrackingLogMessageMon.modules     = cms.vstring(
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
)
FullIterTrackingLogMessageMon.categories     = cms.vstring(
    'TooManyClusters',
    'TooManyPairs',
    'TooManyTriplets',
    'TooManySeeds',
)    

IterTrackingLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
IterTrackingLogMessageMon.pluginsMonName = cms.string ( 'IterTracking' ) 
IterTrackingLogMessageMon.modules     = cms.vstring(
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
)
IterTrackingLogMessageMon.categories     = cms.vstring(
    'TooManyClusters',
    'TooManyPairs',
    'TooManyTriplets',
    'TooManySeeds',
)    


ConversionLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
ConversionLogMessageMon.pluginsMonName = cms.string ( 'Conversion' ) 
ConversionLogMessageMon.modules     = cms.vstring(
       'photonConvTrajSeedFromSingleLeg',
       'convTrackCandidates',
       'convStepTracks',
)
ConversionLogMessageMon.categories     = cms.vstring(
    'TooManyClusters',
    'TooManyPairs',
    'TooManyTriplets',
    'TooManySeeds',
)


