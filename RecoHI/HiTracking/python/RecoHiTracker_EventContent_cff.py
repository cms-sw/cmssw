import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoHiTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hiGlobalPrimTracks_*_*', 
		'keep *_hiSelectedTracks_*_*', 
		'keep *_hiPixel3PrimTracks_*_*', 		
		'keep recoVertexs_hiPixelMedianVertex_*_*',
		'keep recoVertexs_hiPixelAdaptiveVertex_*_*'		
    )
)
#RECO content
RecoHiTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hiGlobalPrimTracks_*_*', 
		'keep *_hiSelectedTracks_*_*', 
		'keep *_hiPixel3PrimTracks_*_*', 		
		'keep recoVertexs_hiPixelMedianVertex_*_*',
		'keep recoVertexs_hiPixelAdaptiveVertex_*_*'		
    )
)
#AOD content
RecoHiTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_hiSelectedTracks_*_*',
		'keep recoVertexs_hiPixelMedianVertex_*_*',
		'keep recoVertexs_hiPixelAdaptiveVertex_*_*'		
    )
)
