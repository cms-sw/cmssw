import FWCore.ParameterSet.Config as cms

#FEVT
RecoMTDFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        )
)

#RECO content
RecoMTDRECO = RecoMTDFEVT.copy()

#AOD content
RecoMTDAOD = RecoMTDFEVT.copy()

_phase2TimingLayer_EventContent = cms.untracked.vstring('keep *_trackExtenderWithMTD_*_*')

from Configuration.Eras.Modifier_phase2_timing_layer_tile_cff import phase2_timing_layer_tile
from Configuration.Eras.Modifier_phase2_timing_layer_bar_cff import phase2_timing_layer_bar

def modifyDataTier(DataTier):
    added = DataTier.outputCommands + _phase2TimingLayer_EventContent
    (phase2_timing_layer_tile | phase2_timing_layer_bar).toModify(DataTier, outputCommands = added)

modifyDataTier(RecoMTDFEVT)
modifyDataTier(RecoMTDRECO)
modifyDataTier(RecoMTDAOD)
