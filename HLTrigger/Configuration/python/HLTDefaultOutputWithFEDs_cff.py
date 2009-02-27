# /dev/CMSSW_3_1_0/pre2/HLT/V123 (CMSSW_3_1_X_2009-02-25-0700_HLT2)

import FWCore.ParameterSet.Config as cms


HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_3_1_0/pre2/HLT/V123')
)

block_hltDefaultOutputWithFEDs = cms.PSet(
outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
  'keep FEDRawDataCollection_rawDataCollector_*_*',
  'keep edmTriggerResults_*_*_*',
  'keep triggerTriggerEvent_*_*_*' )
)
