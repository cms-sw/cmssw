import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
PAtrackingMonHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone(
    beamSpot         = "hltOnlineBeamSpot",
    primaryVertex    = "hltPixelVertices",
    doAllPlots       = False,
    doLumiAnalysis   = False,     
    doProfilesVsLS   = True,
    doDCAPlots       = True,
    doPUmonitoring   = False,
    doPlotsVsGoodPVtx = False,
    doEffFromHitPatternVsPU = False,
    pvNDOF           = 1,
    numCut           = " pt >= 0.4 & quality('highPurity') ",
    denCut           = " pt >= 0.4",
    FolderName       = "TrackingPA/GlobalParameters",
    BSFolderName     = "TrackingPA/ParametersVsBeamSpot"
)
PApixelTracksMonitoringHLT = PAtrackingMonHLT.clone(
    FolderName       = 'HLT/TrackingPA/pixelTracks',
    TrackProducer    = 'hltPixelTracks',
    allTrackProducer = 'hltPixelTracks',
)
PApixelTracksForHighMultMonitoringHLT = PAtrackingMonHLT.clone(
    FolderName       = 'HLT/TrackingPA/pixelTracksForHighMult',
    primaryVertex    = 'hltPixelVerticesForHighMult',
    TrackProducer    = 'hltPixelTracksForHighMult',
    allTrackProducer = 'hltPixelTracksForHighMult'
)
PAiter0TracksMonitoringHLT = PAtrackingMonHLT.clone(
    FolderName       = 'HLT/TrackingPA/iter0',
    primaryVertex    = 'hltPAOnlinePrimaryVertices',
    TrackProducer    = 'hltPAIter0CtfWithMaterialTracks',
    allTrackProducer = 'hltPAIter0CtfWithMaterialTracks'
)
PAiter1TracksMonitoringHLT = PAtrackingMonHLT.clone(
    FolderName       = 'HLT/TrackingPA/iter1',
    primaryVertex    = 'hltPAOnlinePrimaryVertices',
    TrackProducer    = 'hltPAIter1CtfWithMaterialTracks',
    allTrackProducer = 'hltPAIter1CtfWithMaterialTracks'
)
PAiter2TracksMonitoringHLT = PAtrackingMonHLT.clone(
    FolderName       = 'HLT/TrackingPA/iter2',
    primaryVertex    = 'hltPAOnlinePrimaryVertices',
    TrackProducer    = 'hltPAIter2CtfWithMaterialTracks',
    allTrackProducer = 'hltPAIter2CtfWithMaterialTracks'
)
PAiter3TracksMonitoringHLT = PAtrackingMonHLT.clone(
    FolderName       = 'HLT/TrackingPA/iter3',
    primaryVertex    = 'hltPAOnlinePrimaryVertices',
    TrackProducer    = 'hltPAIter3CtfWithMaterialTracks',
    allTrackProducer = 'hltPAIter3CtfWithMaterialTracks'
)
PAiter4TracksMonitoringHLT = PAtrackingMonHLT.clone(
    FolderName       = 'HLT/TrackingPA/iter4',
    primaryVertex    = 'hltPAOnlinePrimaryVertices',
    TrackProducer    = 'hltPAIter4CtfWithMaterialTracks',
    allTrackProducer = 'hltPAIter4CtfWithMaterialTracks'
)
PAiter5TracksMonitoringHLT = PAtrackingMonHLT.clone(
    FolderName       = 'HLT/TrackingPA/iter5',
    primaryVertex    = 'hltPAOnlinePrimaryVertices',
    TrackProducer    = 'hltPAIter5CtfWithMaterialTracks',
    allTrackProducer = 'hltPAIter5CtfWithMaterialTracks'
)
PAiter6TracksMonitoringHLT = PAtrackingMonHLT.clone(
    FolderName       = 'HLT/TrackingPA/iter6',
    primaryVertex    = 'hltPAOnlinePrimaryVertices',
    TrackProducer    = 'hltPAIter6CtfWithMaterialTracks',
    allTrackProducer = 'hltPAIter6CtfWithMaterialTracks'
)
PAiter7TracksMonitoringHLT = PAtrackingMonHLT.clone(
    FolderName       = 'HLT/TrackingPA/iter7',
    primaryVertex    = 'hltPAOnlinePrimaryVertices',
    TrackProducer    = 'hltPAIter7CtfWithMaterialTracks',
    allTrackProducer = 'hltPAIter7CtfWithMaterialTracks'
)
PAiterHLTTracksMonitoringHLT = PAtrackingMonHLT.clone(
    FolderName       = 'HLT/TrackingPA/iterMerged',
    primaryVertex    = 'hltPAOnlinePrimaryVertices',
    TrackProducer    = 'hltPAIterativeTrackingMerged',
    allTrackProducer = 'hltPAIterativeTrackingMerged'
)
PAtrackingMonitorHLT = cms.Sequence(
    PApixelTracksMonitoringHLT
    + PApixelTracksForHighMultMonitoringHLT
    + PAiter0TracksMonitoringHLT
    + PAiter1TracksMonitoringHLT
    + PAiter2TracksMonitoringHLT
    + PAiter3TracksMonitoringHLT
    + PAiter4TracksMonitoringHLT
    + PAiter5TracksMonitoringHLT
    + PAiter6TracksMonitoringHLT
    + PAiter7TracksMonitoringHLT
    + PAiterHLTTracksMonitoringHLT
)
