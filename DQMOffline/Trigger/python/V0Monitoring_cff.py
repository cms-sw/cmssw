import FWCore.ParameterSet.Config as cms

# Monitoring of the HLT V0 candidates (K0s -> pi+ pi-, Lambda -> p pi-)
# reconstructed online by hltGeneralV0Candidates (V0Producer run on hltGeneralTracks)
import DQM.TrackingMonitor.V0Monitor_cff as _v0Monitor_cff

hltKshortMonitoring = _v0Monitor_cff.KshortMonitoring.clone(
    FolderName    = 'HLT/Tracking/V0Monitoring/Ks',
    v0            = 'hltGeneralV0Candidates:Kshort',
    beamSpot      = 'hltOnlineBeamSpot',
    primaryVertex = 'hltPhase2PixelVertices',
    forceSCAL     = False,
    pvNDOF        = 1
)

hltLambdaMonitoring = _v0Monitor_cff.LambdaMonitoring.clone(
    FolderName    = 'HLT/Tracking/V0Monitoring/Lambda',
    v0            = 'hltGeneralV0Candidates:Lambda',
    beamSpot      = 'hltOnlineBeamSpot',
    primaryVertex = 'hltPhase2PixelVertices',
    forceSCAL     = False,
    pvNDOF        = 1
)

v0MonitoringHLT = cms.Sequence(
    hltKshortMonitoring
    + hltLambdaMonitoring
)
