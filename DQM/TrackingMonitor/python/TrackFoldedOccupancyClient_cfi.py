import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

TrackerMapFoldedClient = DQMEDHarvester("TrackFoldedOccupancyClient",  
    FolderName = cms.string('Tracking/TrackParameters'),
    AlgoName = cms.string('GenTk'),
    TrackQuality = cms.string('generalTracks'),
    MeasurementState = cms.string('ImpactPoint'),
    PhiMax = cms.double(3.141592654),
    PhiMin = cms.double(-3.141592654),
    EtaMax = cms.double(2.5),
    EtaMin = cms.double(-2.5),  
    Eta2DBin = cms.int32(26),
    Phi2DBin = cms.int32(32),
) 
    
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(TrackerMapFoldedClient, EtaMin=-3., EtaMax=3.)
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(TrackerMapFoldedClient, EtaMin=-4.5, EtaMax=4.5)

TrackerMapFoldedClient_highpurity_dzPV0p1=TrackerMapFoldedClient.clone(
    TrackQuality=cms.string('highPurityTracks/dzPV0p1')
)

TrackerMapFoldedClient_highpurity_pt0to1=TrackerMapFoldedClient.clone(
    TrackQuality=cms.string('highPurityTracks/pt_0to1')
)

TrackerMapFoldedClient_highpurity_pt1=TrackerMapFoldedClient.clone(
    TrackQuality=cms.string('highPurityTracks/pt_1')
)

foldedMapClientSeq=cms.Sequence(TrackerMapFoldedClient*TrackerMapFoldedClient_highpurity_dzPV0p1*TrackerMapFoldedClient_highpurity_pt0to1*TrackerMapFoldedClient_highpurity_pt1)

#run3
TrackerMapFoldedClient_hiConformalPixelTracks=TrackerMapFoldedClient.clone(
    TrackQuality = cms.string('hiConformalPixelTracks')
)

folded_with_conformalpixtkclient= cms.Sequence(TrackerMapFoldedClient_hiConformalPixelTracks+foldedMapClientSeq.copy())
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(foldedMapClientSeq, folded_with_conformalpixtkclient)
