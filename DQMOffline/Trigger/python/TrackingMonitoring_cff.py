import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
trackingMonHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone(
    beamSpot                = "hltOnlineBeamSpot",
    primaryVertex           = "hltPixelVertices",
    doAllPlots              = False,
    doLumiAnalysis          = False,     
    #doProfilesVsLS          = True,
    doDCAPlots              = True,
    pvNDOF                  = 1,
    doProfilesVsLS          = True,
    #doPlotsVsGoodPVtx       = True,
    doEffFromHitPatternVsPU = True,
    doEffFromHitPatternVsBX = True,
    doEffFromHitPatternVsLUMI = True,
    doPlotsVsGoodPVtx       = True,
    doPlotsVsLUMI           = True,
    doPlotsVsBX             = True
)
pixelTracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/pixelTracks',
    TrackProducer    = 'hltPixelTracks',
    allTrackProducer = 'hltPixelTracks',
    doEffFromHitPatternVsPU   = False,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(pixelTracksMonitoringHLT,
                        TrackProducer    = 'hltPhase2PixelTracks',
                        allTrackProducer = 'hltPhase2PixelTracks')

iter0TracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/iter0',
    TrackProducer    = 'hltIter0PFlowCtfWithMaterialTracks',
    allTrackProducer = 'hltIter0PFlowCtfWithMaterialTracks',
    doEffFromHitPatternVsPU   = True,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False
)
iter0HPTracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/iter0HP',
    TrackProducer    = 'hltIter0PFlowTrackSelectionHighPurity',
    allTrackProducer = 'hltIter0PFlowTrackSelectionHighPurity',
    doEffFromHitPatternVsPU   = True,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False
)
iter1TracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/iter1',
    TrackProducer    = 'hltIter1PFlowCtfWithMaterialTracks',
    allTrackProducer = 'hltIter1PFlowCtfWithMaterialTracks',
    doEffFromHitPatternVsPU   = True,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False
)
iter1HPTracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/iter1HP',
    TrackProducer    = 'hltIter1PFlowTrackSelectionHighPurity',
    allTrackProducer = 'hltIter1PFlowTrackSelectionHighPurity',
    doEffFromHitPatternVsPU   = True,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False
)
iter2TracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/iter2',
    TrackProducer    = 'hltIter2PFlowCtfWithMaterialTracks',
    allTrackProducer = 'hltIter2PFlowCtfWithMaterialTracks',
    doEffFromHitPatternVsPU   = True,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False
)
iter2HPTracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/iter2HP',
    TrackProducer    = 'hltIter2PFlowTrackSelectionHighPurity',
    allTrackProducer = 'hltIter2PFlowTrackSelectionHighPurity',
    doEffFromHitPatternVsPU   = True,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False
)
iter2MergedTracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/iter2Merged',
    TrackProducer    = 'hltIter2Merged',
    allTrackProducer = 'hltIter2Merged',
    doEffFromHitPatternVsPU   = True,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False
)
iterHLTTracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/tracks',
    TrackProducer    = 'hltMergedTracks',
    allTrackProducer = 'hltMergedTracks',
    doEffFromHitPatternVsPU   = True,
    doEffFromHitPatternVsBX   = True,
    doEffFromHitPatternVsLUMI = True,
    doDCAPlots                = True,
    doPVPlots                 = cms.bool(True),
    doBSPlots                 = cms.bool(True),
    doSIPPlots                = cms.bool(True)
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(iterHLTTracksMonitoringHLT,
                        TrackProducer    = cms.InputTag("generalTracks","","HLT"),
                        allTrackProducer = cms.InputTag("generalTracks","","HLT"))

iter3TracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/iter3Merged',
    TrackProducer    = 'hltIter3Merged',
    allTrackProducer = 'hltIter3Merged'
)
iter4TracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/iter4Merged',
    TrackProducer    = 'hltIter4Merged',
    allTrackProducer = 'hltIter4Merged'
)
trackingMonitorHLT = cms.Sequence(
    pixelTracksMonitoringHLT
    + iter0HPTracksMonitoringHLT
#    + iter1HPTracksMonitoringHLT
#    + iter2HPTracksMonitoringHLT
    + iter2MergedTracksMonitoringHLT
    + iterHLTTracksMonitoringHLT
)    

trackingMonitorHLTall = cms.Sequence(
    pixelTracksMonitoringHLT
    + iter0TracksMonitoringHLT
    + iter0HPTracksMonitoringHLT
    + iter1TracksMonitoringHLT
    + iter1HPTracksMonitoringHLT
    + iter2TracksMonitoringHLT
    + iter2HPTracksMonitoringHLT
    + iter2MergedTracksMonitoringHLT
    + iterHLTTracksMonitoringHLT
#    + iter3TracksMonitoringHLT
#    + iter4TracksMonitoringHLT
)    

doubletRecoveryHPTracksMonitoringHLT = trackingMonHLT.clone(
    FolderName       = 'HLT/Tracking/doubletRecoveryTracks',
    TrackProducer    = 'hltDoubletRecoveryPFlowTrackSelectionHighPurity',
    allTrackProducer = 'hltDoubletRecoveryPFlowTrackSelectionHighPurity',
    doEffFromHitPatternVsPU   = True,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False
)

############
#### EGM tracks
# GSF: hltEgammaGsfTracks
# Iter0: process.hltIter0ElectronsTrackSelectionHighPurity
# Iter1HP: hltIter1MergedForElectrons
# Iter2HP: hltIter2MergedForElectrons
egmTrackingMonHLT = trackingMonHLT.clone(
    primaryVertex = "hltElectronsVertex",
    doEffFromHitPatternVsPU   = False,
    doEffFromHitPatternVsBX   = False,
    doEffFromHitPatternVsLUMI = False 
)
gsfTracksMonitoringHLT = egmTrackingMonHLT.clone(
    FolderName       = 'HLT/EGM/Tracking/GSF',
    TrackProducer    = 'hltEgammaGsfTracks',
    allTrackProducer = 'hltEgammaGsfTracks'
)
pixelTracksForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone(
    FolderName       = 'HLT/EGM/Tracking/pixelTracks',
    TrackProducer    = 'hltPixelTracksElectrons',
    allTrackProducer = 'hltPixelTracksElectrons'
)
iter0ForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone(
    FolderName       = 'HLT/EGM/Tracking/iter0',
    TrackProducer    = 'hltIter0ElectronsCtfWithMaterialTracks',
    allTrackProducer = 'hltIter0ElectronsCtfWithMaterialTracks'
)
iter0HPForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone(
    FolderName       = 'HLT/EGM/Tracking/iter0HP',
    TrackProducer    = 'hltIter0ElectronsTrackSelectionHighPurity',
    allTrackProducer = 'hltIter0ElectronsTrackSelectionHighPurity'
)
iter1ForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone(
    FolderName       = 'HLT/EGM/Tracking/iter1',
    TrackProducer    = 'hltIter1ElectronsCtfWithMaterialTracks',
    allTrackProducer = 'hltIter1ElectronsCtfWithMaterialTracks'
)
iter1HPForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone(
    FolderName       = 'HLT/EGM/Tracking/iter1HP',
    TrackProducer    = 'hltIter1ElectronsTrackSelectionHighPurity',
    allTrackProducer = 'hltIter1ElectronsTrackSelectionHighPurity'
)
iter1MergedForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone(
    FolderName       = 'HLT/EGM/Tracking/iter1Merged',
    TrackProducer    = 'hltIter1MergedForElectrons',
    allTrackProducer = 'hltIter1MergedForElectrons'
)
iter2ForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone(
    FolderName       = 'HLT/EGM/Tracking/iter2',
    TrackProducer    = 'hltIter2ElectronsCtfWithMaterialTracks',
    allTrackProducer = 'hltIter2ElectronsCtfWithMaterialTracks'
)
iter2HPForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone(
    FolderName       = 'HLT/EGM/Tracking/iter2HP',
    TrackProducer    = 'hltIter2ElectronsTrackSelectionHighPurity',
    allTrackProducer = 'hltIter2ElectronsTrackSelectionHighPurity'
)
iterHLTTracksForElectronsMonitoringHLT = egmTrackingMonHLT.clone(
    FolderName       = 'HLT/EGM/Tracking/iter2Merged',
    TrackProducer    = 'hltIter2MergedForElectrons',
    allTrackProducer = 'hltIter2MergedForElectrons'
)

egmTrackingMonitorHLT = cms.Sequence(
    gsfTracksMonitoringHLT
    + pixelTracksForElectronsTracksMonitoringHLT
    + iter0HPForElectronsTracksMonitoringHLT
#    + iter1HPForElectronsTracksMonitoringHLT
#    + iter2HPForElectronsTracksMonitoringHLT
    + iterHLTTracksForElectronsMonitoringHLT
)

trkHLTDQMSourceExtra = cms.Sequence(
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toReplaceWith(trackingMonitorHLT, cms.Sequence(pixelTracksMonitoringHLT + iterHLTTracksMonitoringHLT + doubletRecoveryHPTracksMonitoringHLT )) # + iter0HPTracksMonitoringHLT ))
phase2_tracker.toReplaceWith(trackingMonitorHLT, cms.Sequence(pixelTracksMonitoringHLT + iterHLTTracksMonitoringHLT))

run3_common.toReplaceWith(trackingMonitorHLTall, cms.Sequence(pixelTracksMonitoringHLT + iter0TracksMonitoringHLT + iterHLTTracksMonitoringHLT))
run3_common.toReplaceWith(egmTrackingMonitorHLT, cms.Sequence(gsfTracksMonitoringHLT))
