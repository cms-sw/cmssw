# The following comments couldn't be translated into the new config version:

import FWCore.ParameterSet.Config as cms

# AOD content
RecoLocalMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_dt4DSegments_*_*', 
        'keep *_dt4DCosmicSegments_*_*',
        'keep *_cscSegments_*_*',
        'keep *_rpcRecHits_*_*')
)
from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
from Configuration.Eras.Modifier_bParking_cff import bParking
for e in [run2_GEM_2017, run3_GEM]:
    e.toModify( RecoLocalMuonAOD, 
                outputCommands = RecoLocalMuonAOD.outputCommands + [
        'keep *_gemRecHits_*_*', 
        'keep *_gemSegments_*_*'])

phase2_muon.toModify( RecoLocalMuonAOD, 
    outputCommands = RecoLocalMuonAOD.outputCommands + [
        'keep *_me0RecHits_*_*', 
        'keep *_me0Segments_*_*'])
bParking.toModify( RecoLocalMuonAOD, 
    outputCommands = RecoLocalMuonAOD.outputCommands + [
        'keep *_dt1DRecHits_*_*',
        'keep *_csc2DRecHits_*_*'])


# RECO content
RecoLocalMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_dt1DRecHits_*_*', 
        'keep *_dt1DCosmicRecHits_*_*',
        'keep *_csc2DRecHits_*_*')
)
RecoLocalMuonRECO.outputCommands.extend(RecoLocalMuonAOD.outputCommands)

# Full Event content 
RecoLocalMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoLocalMuonFEVT.outputCommands.extend(RecoLocalMuonRECO.outputCommands)
