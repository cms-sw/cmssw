# The following comments couldn't be translated into the new config version:

import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoLocalMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_dt1DRecHits_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*')
)
# RECO content
RecoLocalMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_dt1DRecHits_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_dt1DCosmicRecHits_*_*',
        'keep *_dt4DCosmicSegments_*_*',
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*')
)
# AOD content
RecoLocalMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_dt4DSegments_*_*', 
        'keep *_dt4DCosmicSegments_*_*',
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*')
)

from Configuration.StandardSequences.Eras import eras
if eras.phase2_muon.isChosen() or eras.phase2dev_muon.isChosen():
    RecoLocalMuonFEVT.outputCommands.append('keep *_gemRecHits_*_*')
    RecoLocalMuonFEVT.outputCommands.append('keep *_me0RecHits_*_*')
    RecoLocalMuonFEVT.outputCommands.append('keep *_me0Segments_*_*')

    RecoLocalMuonRECO.outputCommands.append('keep *_gemRecHits_*_*')
    RecoLocalMuonRECO.outputCommands.append('keep *_me0RecHits_*_*')
    RecoLocalMuonRECO.outputCommands.append('keep *_me0Segments_*_*')

    RecoLocalMuonAOD.outputCommands.append('keep *_gemRecHits_*_*')
    RecoLocalMuonAOD.outputCommands.append('keep *_me0RecHits_*_*')
    RecoLocalMuonAOD.outputCommands.append('keep *_me0Segments_*_*')

