import FWCore.ParameterSet.Config as cms


selectedModules = []
selectedModules4cosmics = []

pluginsMonName = {}
modulesLabel        = {}
categories     = {}

### LocalReco
pluginsMonName['LocalReco'] = cms.string ('LocalReco')
modulesLabel  ['LocalReco'] = cms.vstring('siPixelDigis', 'siStripDigis', 'siPixelClusters', 'siStripClusters' ) # siPixelDigis : SiPixelRawToDigi, siStripDigis : SiStripRawToDigi (SiStripRawToDigiUnpacker), siPixelClusters : SiPixelClusterProducer, siStripClusters : SiStripClusterizer
categories    ['LocalReco'] = cms.vstring('SiPixelRawToDigi', 'TooManyErrors', 'TooManyClusters' )


# apparentely there are not LogError in RecoLocalTracker/SubCollectionProducers/src/TrackClusterRemover.cc
pluginsMonName['Clusterizer'] = cms.string ( 'TrackClusterRemover' )
modulesLabel  ['Clusterizer'] = cms.vstring( 'lowPtTripletStepClusters', 'pixelPairStepClusters', 'detachedTripletStepClusters', 'mixedTripletStepClusters', 'pixelLessStepClusters', 'tobTecStepClusters' ) # TrackClusterRemover
categories    ['Clusterizer'] = cms.vstring(  )

# initialStepSeeds,lowPtTripletStepSeeds, pixelPairStepSeeds, detachedTripletStepSeeds, : TooManyClusters (SeedGeneratorFromRegionHitsEDProducer),
# photonConvTrajSeedFromSingleLeg : (PhotonConversionTrajectorySeedProducerFromSingleLeg)
pluginsMonName['Seeding'] = cms.string ( 'Seeding' ) 
modulesLabel  ['Seeding'] = cms.vstring( 'initialStepSeeds', 'lowPtTripletStepSeeds', 'pixelPairStepSeeds', 'detachedTripletStepSeeds', 'mixedTripletStepSeedsA', 'mixedTripletStepSeedsB', 'mixedTripletStepSeeds', 'pixelLessStepSeeds', 'tobTecStepSeeds', 'photonConvTrajSeedFromSingleLeg')
categories    ['Seeding'] = cms.vstring( 'TooManyClusters', 'TooManyPairs', 'TooManyTriplets', 'TooManySeeds' )

# RecoTracker/CkfPattern/src/CkfTrackCandidateMakerBase.cc
pluginsMonName['TrackCandidate'] = cms.string ( 'TrackCandidate' ) 
modulesLabel  ['TrackCandidate'] = cms.vstring( 'initialStepTrackCandidates', 'lowPtTripletStepTrackCandidates', 'pixelPairStepTrackCandidates', 'detachedTripletStepTrackCandidates', 'mixedTripletStepTrackCandidates', 'pixelLessStepTrackCandidates', 'tobTecStepTrackCandidates', 'convTrackCandidates' )
categories    ['TrackCandidate'] = cms.vstring( 'TooManySeeds', 'CkfPattern', 'BaseCkfTrajectoryBuilder_InfiniteLoop' )

# TrackProducer:FailedPropagation 
pluginsMonName['TrackFinder'] = cms.string ( 'TrackFinder' ) 
modulesLabel  ['TrackFinder'] = cms.vstring( 'pixelTracks', 'initialStepTracks', 'lowPtTripletStepTracks', 'pixelPairStepTracks', 'detachedTripletStepTracks', 'mixedTripletStepTracks', 'pixelLessStepTracks', 'tobTecStepTracks', 'generalTracks' )
categories    ['TrackFinder'] = cms.vstring( 'FailedPropagation' )


pluginsMonName['FullIterTracking'] = cms.string ( 'FullIterTracking' ) 
modulesLabel  ['FullIterTracking'] = cms.vstring(
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
categories['FullIterTracking']     = cms.vstring(
    'TooManyClusters',
    'TooManyPairs',
    'TooManyTriplets',
    'TooManySeeds',
)    


pluginsMonName['IterTracking'] = cms.string ( 'IterTracking' ) 
modulesLabel  ['IterTracking'] = cms.vstring(
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
categories['IterTracking']     = cms.vstring(
    'TooManyClusters',
    'TooManyPairs',
    'TooManyTriplets',
    'TooManySeeds',
)    


pluginsMonName['Conversion'] = cms.string ( 'Conversion' ) 
modulesLabel  ['Conversion'] = cms.vstring( 'photonConvTrajSeedFromSingleLeg', 'convTrackCandidates', 'convStepTracks' )
categories    ['Conversion'] = cms.vstring( 'TooManyClusters', 'TooManyPairs', 'TooManyTriplets', 'TooManySeeds' )


selectedModules.extend( ['LocalReco'] )
selectedModules.extend( ['Clusterizer'] )
selectedModules.extend( ['Seeding'] )
selectedModules.extend( ['TrackCandidate'] )
selectedModules.extend( ['TrackFinder'] )

# cosmicseedfinderP5 (CosmicSeedGenerator): TooManyClusters; combinatorialcosmicseedfinderP5 (SeedGeneratorFromRegionHitsEDProducer):TooManyClusters; regionalCosmicTrackerSeeds (CtfSpecialSeedGenerator): TooManyClusters, CtfSpecialSeedGenerator 
pluginsMonName['cosmicsSeeding'] = cms.string ( 'Seeding' ) 
modulesLabel  ['cosmicsSeeding'] = cms.vstring( 'cosmicseedfinderP5', 'combinatorialcosmicseedfinderP5', 'regionalCosmicTrackerSeeds', 'CtfSpecialSeedGenerator' )
categories    ['cosmicsSeeding'] = cms.vstring( 'TooManyClusters' )

# ckfTrackCandidatesP5 (CkfTrackCandidateMaker), regionalCosmicCkfTrackCandidates (CkfTrackCandidateMaker): 
# RecoTracker/CkfPattern/src/BaseCkfTrajectoryBuilder.cc
# - CkfPattern
# - BaseCkfTrajectoryBuilder_InfiniteLoop
# RecoTracker/CkfPattern/src/CkfTrajectoryBuilder.cc
# - CkfPattern
# RecoTracker/CkfPattern/src/CkfTrackCandidateMakerBase.cc
# - TooManySeeds
pluginsMonName['cosmicsTrackCandidate'] = cms.string ( 'TrackCandidate' ) 
modulesLabel  ['cosmicsTrackCandidate'] = cms.vstring( 'ckfTrackCandidatesP5', 'regionalCosmicCkfTrackCandidates' )
categories    ['cosmicsTrackCandidate'] = cms.vstring( 'CkfPattern', 'BaseCkfTrajectoryBuilder_InfiniteLoop', 'TooManySeeds' )

pluginsMonName['cosmicsTrack'] = cms.string ( 'TrackFinder' ) 
modulesLabel  ['cosmicsTrack'] = cms.vstring( 'ctfWithMaterialTracksCosmics', 'regionalCosmicTracks' )
categories    ['cosmicsTrack'] = cms.vstring( 'FailedPropagation', 'RKPropagatorInS' )


selectedModules4cosmics.extend( ['LocalReco'] )
selectedModules4cosmics.extend( ['cosmicsSeeding'] )
selectedModules4cosmics.extend( ['cosmicsTrackCandidate'] )
selectedModules4cosmics.extend( ['cosmicsTrack'] )
