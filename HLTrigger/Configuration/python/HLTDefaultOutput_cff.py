# /dev/CMSSW_2_1_9/HLT/V10 (CMSSW_2_1_X_2008-10-09-0000)

import FWCore.ParameterSet.Config as cms


HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_2_1_9/HLT/V10')
)

block_hltDefaultOutput = cms.PSet(
outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
  'keep edmTriggerResults_*_*_*',
  'keep triggerTriggerEvent_*_*_*',
  'keep *_hltGctDigis_*_*',
  'keep *_hltGtDigis_*_*',
  'keep *_hltL1extraParticles_*_*',
  'keep *_hltL1GtObjectMap_*_*' )
)
