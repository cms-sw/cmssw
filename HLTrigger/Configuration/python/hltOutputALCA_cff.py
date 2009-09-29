# /dev/CMSSW_3_3_0/pre4/HLT/V20 (CMSSW_3_3_X_2009-09-17-0100_HLT3)

import FWCore.ParameterSet.Config as cms


HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_3_3_0/pre4/HLT/V20')
)


block_hltOutputALCAPHISYM = cms.PSet(
outputCommands = cms.untracked.vstring( 'drop *',
  'keep edmTriggerResults_*_*_*',
  'keep triggerTriggerEvent_*_*_*',
  'keep *_hltAlCaPhiSymStream_*_*',
  'keep *_hltGtDigis_*_*' )
)
block_hltOutputALCAPHISYMHCAL = cms.PSet(
outputCommands = cms.untracked.vstring( 'drop *',
  'keep edmTriggerResults_*_*_*',
  'keep triggerTriggerEvent_*_*_*',
  'keep *_hltL1extraParticles_*_*',
  'keep *_hltGctDigis_*_*',
  'keep *_hltAlCaHcalFEDSelector_*_*',
  'keep *_hltGtDigis_*_*',
  'keep *_hltL1GtObjectMap_*_*' )
)
block_hltOutputALCAP0 = cms.PSet(
outputCommands = cms.untracked.vstring( 'drop *',
  'keep edmTriggerResults_*_*_*',
  'keep triggerTriggerEvent_*_*_*',
  'keep *_hltAlCaEtaRegRecHitsCosmics_*_*',
  'keep *_hltAlCaPi0RegRecHitsCosmics_*_*',
  'keep *_hltAlCaPi0RegRecHits_*_*',
  'keep *_hltAlCaEtaRegRecHits_*_*' )
)
block_hltOutputRPCMON = cms.PSet(
outputCommands = cms.untracked.vstring( 'drop *',
  'keep edmTriggerResults_*_*_*',
  'keep *_hltRpcRecHits_*_*',
  'keep *_hltMuonDTDigis_*_*',
  'keep *_hltCscSegments_*_*',
  'keep *_hltDt4DSegments_*_*',
  'keep L1MuGMTCands_hltGtDigis_*_*',
  'keep L1MuGMTReadoutCollection_hltGtDigis_*_*',
  'keep *_hltMuonRPCDigis_*_*' )
)
