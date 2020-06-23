import FWCore.ParameterSet.Config as cms

# AOD content
RecoLocalMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon

run2_GEM_2017.toModify( RecoLocalMuonAOD, 
    outputCommands = RecoLocalMuonAOD.outputCommands + [
	'keep *_gemRecHits_*_*', 
	'keep *_gemSegments_*_*'])
run3_GEM.toModify( RecoLocalMuonAOD, 
    outputCommands = RecoLocalMuonAOD.outputCommands + [
	'keep *_gemRecHits_*_*', 
	'keep *_gemSegments_*_*'])
phase2_muon.toModify( RecoLocalMuonAOD, 
    outputCommands = RecoLocalMuonAOD.outputCommands + [
	'keep *_me0RecHits_*_*', 
	'keep *_me0Segments_*_*'])

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
        'keep RPCDetIdRPCDigiMuonDigiCollection_muonRPCDigis_*_*', 
        'keep *_rpcRecHits_*_*')
)
RecoLocalMuonRECO.outputCommands.extend(RecoLocalMuonAOD.outputCommands)

# Full Event content 
RecoLocalMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*') 
)
RecoLocalMuonFEVT.outputCommands.extend(RecoLocalMuonRECO.outputCommands)
