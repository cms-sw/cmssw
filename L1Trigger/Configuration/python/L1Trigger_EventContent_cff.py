import FWCore.ParameterSet.Config as cms

# RAW content 
L1TriggerRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep  FEDRawDataCollection_rawDataCollector_*_*',
        'keep  FEDRawDataCollection_source_*_*')
)


# RAWDEBUG content 
L1TriggerRAWDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep  FEDRawDataCollection_rawDataCollector_*_*',
        'keep  FEDRawDataCollection_source_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtTriggerMenuLite_*_*',
        'keep *_conditionsInEdm_*_*',
        'keep *_l1extraParticles_*_*')
)

# RECO content
L1TriggerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtTriggerMenuLite_*_*',
        'keep *_conditionsInEdm_*_*',
        'keep *_l1extraParticles_*_*',
        'keep *_l1L1GtObjectMap_*_*',
        'keep L1MuGMTReadoutCollection_gtDigis_*_*',
        'keep L1GctEmCand*_gctDigis_*_*',
        'keep L1GctJetCand*_gctDigis_*_*',
        'keep L1GctEtHad*_gctDigis_*_*',
        'keep L1GctEtMiss*_gctDigis_*_*',
        'keep L1GctEtTotal*_gctDigis_*_*',
        'keep L1GctHtMiss*_gctDigis_*_*',
        'keep L1GctJetCounts*_gctDigis_*_*',
        'keep L1GctHFRingEtSums*_gctDigis_*_*',
        'keep L1GctHFBitCounts*_gctDigis_*_*',
        'keep LumiDetails_lumiProducer_*_*',
        'keep LumiSummary_lumiProducer_*_*')
)


# AOD content
L1TriggerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtTriggerMenuLite_*_*',
        'keep *_conditionsInEdm_*_*',
        'keep *_l1extraParticles_*_*',
        'keep *_l1L1GtObjectMap_*_*',
        'keep LumiSummary_lumiProducer_*_*')
)


L1TriggerFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_simCscTriggerPrimitiveDigis_*_*', 
        'keep *_simDtTriggerPrimitiveDigis_*_*', 
        'keep *_simRpcTriggerDigis_*_*', 
        'keep *_simRctDigis_*_*', 
        'keep *_simCsctfDigis_*_*', 
        'keep *_simCsctfTrackDigis_*_*', 
        'keep *_simDttfDigis_*_*', 
        'keep *_simGctDigis_*_*', 
        'keep *_simCaloStage1Digis_*_*', 
        'keep *_simCaloStage1FinalDigis_*_*', 
        'keep *_simCaloStage2Layer1Digis_*_*', 
        'keep *_simCaloStage2Digis_*_*', 
        'keep *_simGmtDigis_*_*', 
        'keep *_simGtDigis_*_*', 
        'keep *_cscTriggerPrimitiveDigis_*_*', 
        'keep *_dtTriggerPrimitiveDigis_*_*', 
        'keep *_rpcTriggerDigis_*_*', 
        'keep *_rctDigis_*_*', 
        'keep *_csctfDigis_*_*', 
        'keep *_csctfTrackDigis_*_*', 
        'keep *_dttfDigis_*_*', 
        'keep *_gctDigis_*_*', 
        'keep *_gmtDigis_*_*', 
        'keep *_gtDigis_*_*',
        'keep *_gtEvmDigis_*_*',
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtTriggerMenuLite_*_*',
        'keep *_conditionsInEdm_*_*',
        'keep *_l1extraParticles_*_*',
        'keep *_l1L1GtObjectMap_*_*',
        'keep LumiDetails_lumiProducer_*_*',
        'keep LumiSummary_lumiProducer_*_*')
)

