import FWCore.ParameterSet.Config as cms

# Full Event content 
L1TriggerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*')
)
# RECO content
L1TriggerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*')
)
# AOD content
L1TriggerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*')
)
L1TriggerFEVTDIGI = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_cscTriggerPrimitiveDigis_*_*', 'keep *_dtTriggerPrimitiveDigis_*_*', 'keep *_rpcTriggerDigis_*_*', 'keep *_rctDigis_*_*', 'keep *_csctfDigis_*_*', 'keep *_dttfDigis_*_*', 'keep *_gctDigis_*_*', 'keep *_gmtDigis_*_*', 'keep *_gtDigis_*_*')
)

