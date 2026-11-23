import FWCore.ParameterSet.Config as cms

# Muon collection for Particle Flow and TICL.
from ..modules.hltPhase2L3MuonsNoID_cfi import hltPhase2L3MuonsNoID as _hltPhase2L3MuonsNoID
hltPhase2MuonsForPF = _hltPhase2L3MuonsNoID.clone(
    inputCollectionLabels = ["hltGeneralTracks", "hltPhase2MuonLinksForPF", "hltL2MuonsFromL1TkMuon:UpdatedAtVtx"],
    inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
)
