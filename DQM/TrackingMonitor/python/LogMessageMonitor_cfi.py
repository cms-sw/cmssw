import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.BXlumiParameters_cfi import BXlumiSetup

LogMessageMon = cms.EDAnalyzer("LogMessageMonitor",

    # input modules                               
    modules = cms.vstring(
       'siPixelDigis',
       'siStripDigis',
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
    doWarningsPlots     = cms.bool(False),

    LogFolderName       = cms.string('Tracking/MessageLog'),
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName      = cms.string('MonitorTrack.root'),
    BXlumiSetup         = BXlumiSetup.clone()

## temporary solution
#    BXlumiBin = cms.int32(100), # (400)
#    BXlumiMin = cms.double(1),  # (2000)
#    BXlumiMax = cms.double(10), # (6000)

)    
