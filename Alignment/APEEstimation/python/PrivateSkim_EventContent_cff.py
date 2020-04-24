import FWCore.ParameterSet.Config as cms


SimulationEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
	'drop  *_*_*_RAW',
	'keep  PSimHits_g4SimHits_TrackerHits*_RAW',
	'keep  FEDRawDataCollection_rawDataCollector__RAW',
    ),
    splitLevel = cms.untracked.int32(0),
)



ReconstructionEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
	'drop *_*_*_RAW',
	'drop *_*_*_RECO',
	'keep *_generalTracks_*_RECO',
	'drop Trajectory*_generalTracks_*_RECO',
	'keep Si*ClusteredmNewDetSetVector_*_*_RECO',
    ),
    splitLevel = cms.untracked.int32(0),
)



ApeSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
	'drop *',
	#'keep L1*_*_*_*',
        #'drop *_L1T1*_*_*',
	'keep *_MuSkim_*_*',
        'keep edmTriggerResults_*_*_*'
	#'keep Si*ClusteredmNewDetSetVector_*_*_*',
	
	#'drop *_ALCARECOTkAlMuonIsolated_*_*',
	
	
	#'drop *_*_*_RECO',
        #'keep L1*_*_*_RECO',
	#'drop *_L1T1*_*_*',
	#'drop *_MEtoEDMConverter_*_*',
    ),
)
