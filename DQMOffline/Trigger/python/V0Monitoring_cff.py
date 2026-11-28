import FWCore.ParameterSet.Config as cms

# Monitoring of the HLT V0 candidates (K0s -> pi+ pi-, Lambda -> p pi-)
# reconstructed online by hltGeneralV0Candidates (V0Producer run on hltGeneralTracks)
import DQM.TrackingMonitor.V0Monitor_cff as _v0Monitor_cff

# the number of primary vertices at phase-2 pileup (up to 200) is well beyond the default 0-100 axis
_phase2PUPSet = dict(nbins = 300, xmin = -0.5, xmax = 299.5)

hltKshortMonitoring = _v0Monitor_cff.KshortMonitoring.clone(
    FolderName    = 'HLT/Tracking/V0Monitoring/Ks',
    v0            = 'hltGeneralV0Candidates:Kshort',
    beamSpot      = 'hltOnlineBeamSpot',
    primaryVertex = 'hltPhase2PixelVertices',
    forceSCAL     = False,
    pvNDOF        = 1,
    histoPSet     = dict(puPSet = _phase2PUPSet)
)

hltLambdaMonitoring = _v0Monitor_cff.LambdaMonitoring.clone(
    FolderName    = 'HLT/Tracking/V0Monitoring/Lambda',
    v0            = 'hltGeneralV0Candidates:Lambda',
    beamSpot      = 'hltOnlineBeamSpot',
    primaryVertex = 'hltPhase2PixelVertices',
    forceSCAL     = False,
    pvNDOF        = 1,
    histoPSet     = dict(puPSet = _phase2PUPSet)
)

# hltGeneralV0Candidates is only produced by the phase-2 NGT scouting menu:
# the sequence is empty unless both the phase2_common era and the ngtScouting process modifier are active
v0MonitoringHLT = cms.Sequence()

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
(phase2_common & ngtScouting).toReplaceWith(v0MonitoringHLT, cms.Sequence(
    hltKshortMonitoring
    + hltLambdaMonitoring
))
