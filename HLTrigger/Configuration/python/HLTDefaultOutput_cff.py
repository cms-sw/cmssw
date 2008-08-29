# /dev/CMSSW_2_1_2/HLT/V65 (CMSSW_2_1_X_2008-08-15-0100)

import FWCore.ParameterSet.Config as cms

block_hltDefaultOutput = cms.PSet(
outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*' )
)
