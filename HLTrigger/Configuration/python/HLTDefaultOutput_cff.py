# /dev/CMSSW_2_1_0_pre6/HLT/V9 (CMSSW_2_1_X_2008-06-24-0000_HLT1)

import FWCore.ParameterSet.Config as cms

block_hltDefaultOutput = cms.PSet(
outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*', 'keep FEDRawDataCollection_source_*_*', 'keep FEDRawDataCollection_rawDataCollector_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*' )
)



