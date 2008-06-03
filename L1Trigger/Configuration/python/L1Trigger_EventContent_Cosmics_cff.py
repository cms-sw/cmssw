import FWCore.ParameterSet.Config as cms

# Full Event content 
L1TriggerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_cscTriggerPrimitiveDigis_*_*', 
        'keep *_dtTriggerPrimitiveDigis_*_*', 
        'keep *_rpcTriggerDigis_*_*', 
        'keep *_rctDigis_*_*', 
        'keep *_csctfDigis_*_*', 
        'keep *_csctfTrackDigis_*_*', 
        'keep *_dttfDigis_*_*', 
        'keep *_gctDigis_*_*', 
        'keep *_gmtDigis_*_*', 
        'keep *_gtDigis_*_*', 
        'keep *_valCscTriggerPrimitiveDigis_*_*', 
        'keep *_valDtTriggerPrimitiveDigis_*_*', 
        'keep *_valRpcTriggerDigis_*_*', 
        'keep *_valRctDigis_*_*', 
        'keep *_valCsctfDigis_*_*', 
        'keep *_valCsctfTrackDigis_*_*', 
        'keep *_valDttfDigis_*_*', 
        'keep *_valGctDigis_*_*', 
        'keep *_valGmtDigis_*_*', 
        'keep *_valGtDigis_*_*', 
        'keep *_gtDigis_*_*', 
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtObjectMap_*_*', 
        'keep *_l1extraParticles_*_*', 
        'keep *_l1compare_*_*')
)

