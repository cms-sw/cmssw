import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoHiTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_globalPrimTracks_*_*', 
		'keep *_selectHiTracks_*_*', 
		'keep *_pixel3PrimTracks_*_*',
		'keep recoVertexs_pixel3Vertices_*_*'
    )
)
#RECO content
RecoHiTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_globalPrimTracks_*_*', 
		'keep *_selectHiTracks_*_*', 
		'keep *_pixel3PrimTracks_*_*',		
		'keep recoVertexs_pixel3Vertices_*_*'
    )
)
#AOD content
RecoHiTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_selectHiTracks_*_*',
		'keep recoVertexs_pixel3Vertices_*_*'	
    )
)
