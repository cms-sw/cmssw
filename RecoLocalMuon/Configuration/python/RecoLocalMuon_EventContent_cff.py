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

def _modifyRecoLocalMuonEventContentForRun3( object ):
    object.outputCommands.append('keep *_gemRecHits_*_*')

def _modifyRecoLocalMuonEventContentForPhase2( object ):
    object.outputCommands.append('keep *_me0RecHits_*_*')
    object.outputCommands.append('keep *_me0Segments_*_*')

from Configuration.StandardSequences.Eras import eras
eras.run3_GEM.toModify( RecoLocalMuonFEVT, func=_modifyRecoLocalMuonEventContentForRun3 )
eras.run3_GEM.toModify( RecoLocalMuonRECO, func=_modifyRecoLocalMuonEventContentForRun3 )
eras.run3_GEM.toModify( RecoLocalMuonAOD,  func=_modifyRecoLocalMuonEventContentForRun3 )

eras.phase2_muon.toModify( RecoLocalMuonFEVT, func=_modifyRecoLocalMuonEventContentForPhase2 )
eras.phase2_muon.toModify( RecoLocalMuonRECO, func=_modifyRecoLocalMuonEventContentForPhase2 )
eras.phase2_muon.toModify( RecoLocalMuonAOD,  func=_modifyRecoLocalMuonEventContentForPhase2 )
