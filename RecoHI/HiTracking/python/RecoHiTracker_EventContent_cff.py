import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoHiTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
		'keep *_hiGeneralTracks_*_*', 
                'keep *_hiGeneralAndPixelTracks_*_*',
		'keep *_hiPixel3PrimTracks_*_*', 
		'keep *_hiPixel3ProtoTracks_*_*',	
		'keep *_hiSelectedProtoTracks_*_*',	
		'keep recoVertexs_hiPixelMedianVertex_*_*',
		'keep recoVertexs_hiPixelAdaptiveVertex_*_*',
		'keep recoVertexs_hiSelectedVertex_*_*',
                'keep recoVertexs_hiPixelClusterVertex_*_*'	
    )
)

RecoHiTrackerLocalFEVT = cms.PSet(
   outputCommands = cms.untracked.vstring(
   'keep *_*_APVCM_*',
   'keep *_siStripZeroSuppression_BADAPVBASELINE_*',
   'keep SiStripRawDigiedmDetSetVector_siStripZeroSuppression_VirginRaw_*'
   )
)

#RECO content
RecoHiTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
		'keep *_hiGeneralTracks_*_*', 
                'keep *_hiGeneralAndPixelTracks_*_*',
		'keep recoVertexs_hiPixelMedianVertex_*_*',  
		'keep recoVertexs_hiPixelAdaptiveVertex_*_*',  
		'keep recoVertexs_hiSelectedVertex_*_*',
                'keep recoVertexs_hiPixelClusterVertex_*_*'		
    )
)

RecoHiTrackerLocalRECO = cms.PSet(
   outputCommands = cms.untracked.vstring(
   'keep *_*_APVCM_*'
   #'keep *_siStripZeroSuppression_BADAPVBASELINE_*',
   #'keep SiStripRawDigiedmDetSetVector_siStripZeroSuppression_VirginRaw_*'
   )
)

#AOD content
RecoHiTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_hiGeneralTracks_*_*',
                                           'keep recoTracks_hiGeneralAndPixelTracks_*_*',
                                           'keep recoVertexs_hiSelectedVertex_*_*'		
    )
)
