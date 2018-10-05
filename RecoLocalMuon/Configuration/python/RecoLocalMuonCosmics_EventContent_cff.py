# The following comments couldn't be translated into the new config version:

# DT 

# no Drift algo

# CSC

# RPC

# DT

# no Drift algo

# CSC

# RPC

import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoLocalMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_muonDTDigis_*_*', 
        'keep *_dttfDigis_*_*', 
        'keep *_dt1DRecHits_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_dt4DSegmentsT0Seg_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*', 
        'keep *_rpcRecHits_*_*')
)
# RECO content
RecoLocalMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_muonDTDigis_*_*', 
        'keep *_dttfDigis_*_*', 
        'keep *_dt1DRecHits_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_dt4DSegmentsT0Seg_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*',
        #'keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*', 
        'keep RPCDetIdRPCDigiMuonDigiCollection_muonRPCDigis_*_*', 
        'keep *_rpcRecHits_*_*')
)
# AOD content
RecoLocalMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

def _updateOutput( era, outputPSets, commands):
   for o in outputPSets:
      era.toModify( o, outputCommands = o.outputCommands + commands )

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
_outputs = [RecoLocalMuonFEVT, RecoLocalMuonRECO, RecoLocalMuonAOD]
_updateOutput( run2_GEM_2017, _outputs, ['keep *_gemRecHits_*_*', 'keep *_gemSegments_*_*'] )
_updateOutput( run3_GEM, _outputs, ['keep *_gemRecHits_*_*', 'keep *_gemSegments_*_*'] )
_updateOutput(phase2_muon, _outputs, ['keep *_me0RecHits_*_*', 'keep *_me0Segments_*_*'])

# RPC New Readout Validation
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
_updateOutput(stage2L1Trigger, _outputs, ['keep *_rpcNewRecHits_*_*'])
