import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
trackingMonHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
trackingMonHLT.beamSpot                = cms.InputTag("hltOnlineBeamSpot")
trackingMonHLT.primaryVertex           = cms.InputTag("hltPixelVertices")
trackingMonHLT.doAllPlots              = cms.bool(False)
trackingMonHLT.doLumiAnalysis          = cms.bool(False)     
trackingMonHLT.doProfilesVsLS          = cms.bool(True)
trackingMonHLT.doDCAPlots              = cms.bool(True)
trackingMonHLT.pvNDOF                  = cms.int32(1)
trackingMonHLT.doProfilesVsLS          = cms.bool(True)
trackingMonHLT.doPlotsVsGoodPVtx       = cms.bool(True)
trackingMonHLT.doEffFromHitPatternVsPU = cms.bool(True)
trackingMonHLT.doEffFromHitPatternVsBX = cms.bool(True)
trackingMonHLT.doEffFromHitPatternVsLUMI = cms.bool(True)
trackingMonHLT.doPlotsVsGoodPVtx       = cms.bool(True)
trackingMonHLT.doPlotsVsLUMI           = cms.bool(True)
trackingMonHLT.doPlotsVsBX             = cms.bool(True)

pixelTracksMonitoringHLT = trackingMonHLT.clone()
pixelTracksMonitoringHLT.FolderName       = 'HLT/Tracking/pixelTracks'
pixelTracksMonitoringHLT.TrackProducer    = 'hltPixelTracks'
pixelTracksMonitoringHLT.allTrackProducer = 'hltPixelTracks'

iter0TracksMonitoringHLT = trackingMonHLT.clone()
iter0TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter0'
iter0TracksMonitoringHLT.TrackProducer    = 'hltIter0PFlowCtfWithMaterialTracks'
iter0TracksMonitoringHLT.allTrackProducer = 'hltIter0PFlowCtfWithMaterialTracks'

iter0HPTracksMonitoringHLT = trackingMonHLT.clone()
iter0HPTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter0HP'
iter0HPTracksMonitoringHLT.TrackProducer    = 'hltIter0PFlowTrackSelectionHighPurity'
iter0HPTracksMonitoringHLT.allTrackProducer = 'hltIter0PFlowTrackSelectionHighPurity'

iter1TracksMonitoringHLT = trackingMonHLT.clone()
iter1TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter1'
iter1TracksMonitoringHLT.TrackProducer    = 'hltIter1PFlowCtfWithMaterialTracks'
iter1TracksMonitoringHLT.allTrackProducer = 'hltIter1PFlowCtfWithMaterialTracks'

iter1HPTracksMonitoringHLT = trackingMonHLT.clone()
iter1HPTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter1HP'
iter1HPTracksMonitoringHLT.TrackProducer    = 'hltIter1PFlowTrackSelectionHighPurity'
iter1HPTracksMonitoringHLT.allTrackProducer = 'hltIter1PFlowTrackSelectionHighPurity'

iter2TracksMonitoringHLT = trackingMonHLT.clone()
iter2TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter2'
iter2TracksMonitoringHLT.TrackProducer    = 'hltIter2PFlowCtfWithMaterialTracks'
iter2TracksMonitoringHLT.allTrackProducer = 'hltIter2PFlowCtfWithMaterialTracks'

iter2HPTracksMonitoringHLT = trackingMonHLT.clone()
iter2HPTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter2HP'
iter2HPTracksMonitoringHLT.TrackProducer    = 'hltIter2PFlowTrackSelectionHighPurity'
iter2HPTracksMonitoringHLT.allTrackProducer = 'hltIter2PFlowTrackSelectionHighPurity'

iterHLTTracksMonitoringHLT = trackingMonHLT.clone()
iterHLTTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter2Merged'
iterHLTTracksMonitoringHLT.TrackProducer    = 'hltIter2Merged'
iterHLTTracksMonitoringHLT.allTrackProducer = 'hltIter2Merged'

iter3TracksMonitoringHLT = trackingMonHLT.clone()
iter3TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter3Merged'
iter3TracksMonitoringHLT.TrackProducer    = 'hltIter3Merged'
iter3TracksMonitoringHLT.allTrackProducer = 'hltIter3Merged'

iter4TracksMonitoringHLT = trackingMonHLT.clone()
iter4TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter4Merged'
iter4TracksMonitoringHLT.TrackProducer    = 'hltIter4Merged'
iter4TracksMonitoringHLT.allTrackProducer = 'hltIter4Merged'

trackingMonitorHLT = cms.Sequence(
    pixelTracksMonitoringHLT
    + iter0HPTracksMonitoringHLT
#    + iter1HPTracksMonitoringHLT
#    + iter2HPTracksMonitoringHLT
    + iterHLTTracksMonitoringHLT
)    

trackingMonitorHLTall = cms.Sequence(
    pixelTracksMonitoringHLT
    + iter0TracksMonitoringHLT
    + iter2HPTracksMonitoringHLT
    + iter1TracksMonitoringHLT
    + iter1HPTracksMonitoringHLT
    + iter2TracksMonitoringHLT
    + iter2HPTracksMonitoringHLT
    + iterHLTTracksMonitoringHLT
#    + iter3TracksMonitoringHLT
#    + iter4TracksMonitoringHLT
)    

############
#### EGM tracks
# GSF: hltEgammaGsfTracks
# Iter0: process.hltIter0ElectronsTrackSelectionHighPurity
# Iter1HP: hltIter1MergedForElectrons
# Iter2HP: hltIter2MergedForElectrons
egmTrackingMonHLT = trackingMonHLT.clone()
egmTrackingMonHLT.primaryVertex = cms.InputTag("hltElectronsVertex")

gsfTracksMonitoringHLT = egmTrackingMonHLT.clone()
gsfTracksMonitoringHLT.FolderName       = 'HLT/EG/Tracking/GSF'
gsfTracksMonitoringHLT.TrackProducer    = 'hltEgammaGsfTracks'
gsfTracksMonitoringHLT.allTrackProducer = 'hltEgammaGsfTracks'

pixelTracksForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone()
pixelTracksForElectronsTracksMonitoringHLT.FolderName       = 'HLT/EG/Tracking/pixelTracks'
pixelTracksForElectronsTracksMonitoringHLT.TrackProducer    = 'hltPixelTracksElectrons'
pixelTracksForElectronsTracksMonitoringHLT.allTrackProducer = 'hltPixelTracksElectrons'

iter0ForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone()
iter0ForElectronsTracksMonitoringHLT.FolderName       = 'HLT/EG/Tracking/iter0'
iter0ForElectronsTracksMonitoringHLT.TrackProducer    = 'hltIter0ElectronsCtfWithMaterialTracks'
iter0ForElectronsTracksMonitoringHLT.allTrackProducer = 'hltIter0ElectronsCtfWithMaterialTracks'

iter0HPForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone()
iter0HPForElectronsTracksMonitoringHLT.FolderName       = 'HLT/EG/Tracking/iter0HP'
iter0HPForElectronsTracksMonitoringHLT.TrackProducer    = 'hltIter0ElectronsTrackSelectionHighPurity'
iter0HPForElectronsTracksMonitoringHLT.allTrackProducer = 'hltIter0ElectronsTrackSelectionHighPurity'

iter1ForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone()
iter1ForElectronsTracksMonitoringHLT.FolderName       = 'HLT/EG/Tracking/iter1'
iter1ForElectronsTracksMonitoringHLT.TrackProducer    = 'hltIter1ElectronsCtfWithMaterialTracks'
iter1ForElectronsTracksMonitoringHLT.allTrackProducer = 'hltIter1ElectronsCtfWithMaterialTracks'

iter1HPForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone()
iter1HPForElectronsTracksMonitoringHLT.FolderName       = 'HLT/EG/Tracking/iter1HP'
iter1HPForElectronsTracksMonitoringHLT.TrackProducer    = 'hltIter1ElectronsTrackSelectionHighPurity'
iter1HPForElectronsTracksMonitoringHLT.allTrackProducer = 'hltIter1ElectronsTrackSelectionHighPurity'

iter1MergedForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone()
iter1MergedForElectronsTracksMonitoringHLT.FolderName       = 'HLT/EG/Tracking/iter1Merged'
iter1MergedForElectronsTracksMonitoringHLT.TrackProducer    = 'hltIter1MergedForElectrons'
iter1MergedForElectronsTracksMonitoringHLT.allTrackProducer = 'hltIter1MergedForElectrons'

iter2ForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone()
iter2ForElectronsTracksMonitoringHLT.FolderName       = 'HLT/EG/Tracking/iter2'
iter2ForElectronsTracksMonitoringHLT.TrackProducer    = 'hltIter2ElectronsCtfWithMaterialTracks'
iter2ForElectronsTracksMonitoringHLT.allTrackProducer = 'hltIter2ElectronsCtfWithMaterialTracks'

iter2HPForElectronsTracksMonitoringHLT = egmTrackingMonHLT.clone()
iter2HPForElectronsTracksMonitoringHLT.FolderName       = 'HLT/EG/Tracking/iter2HP'
iter2HPForElectronsTracksMonitoringHLT.TrackProducer    = 'hltIter2ElectronsTrackSelectionHighPurity'
iter2HPForElectronsTracksMonitoringHLT.allTrackProducer = 'hltIter2ElectronsTrackSelectionHighPurity'

iterHLTTracksForElectronsMonitoringHLT = egmTrackingMonHLT.clone()
iterHLTTracksForElectronsMonitoringHLT.FolderName       = 'HLT/EG/Tracking/iter2Merged'
iterHLTTracksForElectronsMonitoringHLT.TrackProducer    = 'hltIter2MergedForElectrons'
iterHLTTracksForElectronsMonitoringHLT.allTrackProducer = 'hltIter2MergedForElectrons'
 

egmTrackingMonitorHLT = cms.Sequence(
    gsfTracksMonitoringHLT
    + pixelTracksForElectronsTracksMonitoringHLT
    + iter0HPForElectronsTracksMonitoringHLT
#    + iter1HPForElectronsTracksMonitoringHLT
#    + iter2HPForElectronsTracksMonitoringHLT
    + iterHLTTracksForElectronsMonitoringHLT
)
