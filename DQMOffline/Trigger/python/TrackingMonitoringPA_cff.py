import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
PAtrackingMonHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
PAtrackingMonHLT.beamSpot         = cms.InputTag("hltOnlineBeamSpot")
PAtrackingMonHLT.primaryVertex    = cms.InputTag("hltPixelVertices")
PAtrackingMonHLT.doAllPlots       = cms.bool(False)
PAtrackingMonHLT.doLumiAnalysis   = cms.bool(False)     
PAtrackingMonHLT.doProfilesVsLS   = cms.bool(True)
PAtrackingMonHLT.doDCAPlots       = cms.bool(True)
PAtrackingMonHLT.doPUmonitoring   = cms.bool(False)
PAtrackingMonHLT.doPlotsVsGoodPVtx = cms.bool(False)
PAtrackingMonHLT.doEffFromHitPatternVsPU = cms.bool(False)
PAtrackingMonHLT.pvNDOF           = cms.int32(1)
PAtrackingMonHLT.numCut           = cms.string(" pt >= 0.4 & quality('highPurity') ")
PAtrackingMonHLT.denCut           = cms.string(" pt >= 0.4")
PAtrackingMonHLT.FolderName       = cms.string("TrackingPA/GlobalParameters")
PAtrackingMonHLT.BSFolderName     = cms.string("TrackingPA/ParametersVsBeamSpot")

PApixelTracksMonitoringHLT = PAtrackingMonHLT.clone()
PApixelTracksMonitoringHLT.FolderName       = 'HLT/TrackingPA/pixelTracks'
PApixelTracksMonitoringHLT.TrackProducer    = 'hltPixelTracks'
PApixelTracksMonitoringHLT.allTrackProducer = 'hltPixelTracks'

PApixelTracksForHighMultMonitoringHLT = PAtrackingMonHLT.clone()
PApixelTracksForHighMultMonitoringHLT.FolderName       = 'HLT/TrackingPA/pixelTracksForHighMult'
PApixelTracksForHighMultMonitoringHLT.primaryVertex    = 'hltPixelVerticesForHighMult'
PApixelTracksForHighMultMonitoringHLT.TrackProducer    = 'hltPixelTracksForHighMult'
PApixelTracksForHighMultMonitoringHLT.allTrackProducer = 'hltPixelTracksForHighMult'

PAiter0TracksMonitoringHLT = PAtrackingMonHLT.clone()
PAiter0TracksMonitoringHLT.FolderName       = 'HLT/TrackingPA/iter0'
PAiter0TracksMonitoringHLT.primaryVertex    = 'hltPAOnlinePrimaryVertices'
PAiter0TracksMonitoringHLT.TrackProducer    = 'hltPAIter0CtfWithMaterialTracks'
PAiter0TracksMonitoringHLT.allTrackProducer = 'hltPAIter0CtfWithMaterialTracks'

PAiter1TracksMonitoringHLT = PAtrackingMonHLT.clone()
PAiter1TracksMonitoringHLT.FolderName       = 'HLT/TrackingPA/iter1'
PAiter1TracksMonitoringHLT.primaryVertex    = 'hltPAOnlinePrimaryVertices'
PAiter1TracksMonitoringHLT.TrackProducer    = 'hltPAIter1CtfWithMaterialTracks'
PAiter1TracksMonitoringHLT.allTrackProducer = 'hltPAIter1CtfWithMaterialTracks'

PAiter2TracksMonitoringHLT = PAtrackingMonHLT.clone()
PAiter2TracksMonitoringHLT.FolderName       = 'HLT/TrackingPA/iter2'
PAiter2TracksMonitoringHLT.primaryVertex    = 'hltPAOnlinePrimaryVertices'
PAiter2TracksMonitoringHLT.TrackProducer    = 'hltPAIter2CtfWithMaterialTracks'
PAiter2TracksMonitoringHLT.allTrackProducer = 'hltPAIter2CtfWithMaterialTracks'

PAiter3TracksMonitoringHLT = PAtrackingMonHLT.clone()
PAiter3TracksMonitoringHLT.FolderName       = 'HLT/TrackingPA/iter3'
PAiter3TracksMonitoringHLT.primaryVertex    = 'hltPAOnlinePrimaryVertices'
PAiter3TracksMonitoringHLT.TrackProducer    = 'hltPAIter3CtfWithMaterialTracks'
PAiter3TracksMonitoringHLT.allTrackProducer = 'hltPAIter3CtfWithMaterialTracks'

PAiter4TracksMonitoringHLT = PAtrackingMonHLT.clone()
PAiter4TracksMonitoringHLT.FolderName       = 'HLT/TrackingPA/iter4'
PAiter4TracksMonitoringHLT.primaryVertex    = 'hltPAOnlinePrimaryVertices'
PAiter4TracksMonitoringHLT.TrackProducer    = 'hltPAIter4CtfWithMaterialTracks'
PAiter4TracksMonitoringHLT.allTrackProducer = 'hltPAIter4CtfWithMaterialTracks'

PAiter5TracksMonitoringHLT = PAtrackingMonHLT.clone()
PAiter5TracksMonitoringHLT.FolderName       = 'HLT/TrackingPA/iter5'
PAiter5TracksMonitoringHLT.primaryVertex    = 'hltPAOnlinePrimaryVertices'
PAiter5TracksMonitoringHLT.TrackProducer    = 'hltPAIter5CtfWithMaterialTracks'
PAiter5TracksMonitoringHLT.allTrackProducer = 'hltPAIter5CtfWithMaterialTracks'

PAiter6TracksMonitoringHLT = PAtrackingMonHLT.clone()
PAiter6TracksMonitoringHLT.FolderName       = 'HLT/TrackingPA/iter6'
PAiter6TracksMonitoringHLT.primaryVertex    = 'hltPAOnlinePrimaryVertices'
PAiter6TracksMonitoringHLT.TrackProducer    = 'hltPAIter6CtfWithMaterialTracks'
PAiter6TracksMonitoringHLT.allTrackProducer = 'hltPAIter6CtfWithMaterialTracks'

PAiter7TracksMonitoringHLT = PAtrackingMonHLT.clone()
PAiter7TracksMonitoringHLT.FolderName       = 'HLT/TrackingPA/iter7'
PAiter7TracksMonitoringHLT.primaryVertex    = 'hltPAOnlinePrimaryVertices'
PAiter7TracksMonitoringHLT.TrackProducer    = 'hltPAIter7CtfWithMaterialTracks'
PAiter7TracksMonitoringHLT.allTrackProducer = 'hltPAIter7CtfWithMaterialTracks'

PAiterHLTTracksMonitoringHLT = PAtrackingMonHLT.clone()
PAiterHLTTracksMonitoringHLT.FolderName       = 'HLT/TrackingPA/iterMerged'
PAiterHLTTracksMonitoringHLT.primaryVertex    = 'hltPAOnlinePrimaryVertices'
PAiterHLTTracksMonitoringHLT.TrackProducer    = 'hltPAIterativeTrackingMerged'
PAiterHLTTracksMonitoringHLT.allTrackProducer = 'hltPAIterativeTrackingMerged'

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
