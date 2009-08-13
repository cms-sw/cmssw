# /dev/CMSSW_3_2_4/HLT/V2 (CMSSW_3_2_4)

import FWCore.ParameterSet.Config as cms


HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_3_2_4/HLT/V2')
)

block_hltDefaultOutput = cms.PSet(
outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
  'keep edmTriggerResults_*_*_*',
  'keep triggerTriggerEvent_*_*_*',
  'keep *_hltL1GtObjectMap_*_*' )
)
