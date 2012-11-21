import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.LogMessageMonitor_cfi

LocalRecoLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
LocalRecoLogMessageMon.modules =  cms.vstring( 'siPixelDigis', 'siStripDigis' )


FullIterTrackingLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
FullIterTrackingLogMessageMon.modules = cms.vstring(
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

IterTrackingLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
IterTrackingLogMessageMon.modules = cms.vstring(
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


ConversionLogMessageMon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
ConversionLogMessageMon.modules = cms.vstring(
       'photonConvTrajSeedFromSingleLeg',
       'convTrackCandidates',
       'convStepTracks',
)
