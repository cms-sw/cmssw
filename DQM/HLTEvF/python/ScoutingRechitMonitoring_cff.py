import FWCore.ParameterSet.Config as cms

## See DQMOffline/HLTScouting/python/HLTScoutingDqmOffline_cff.py

from HLTriggerOffline.Scouting.ScoutingRecHitAnalyzers_cff import *

################################################
# unpack and pack back the L1T results
################################################
from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import gtStage2Digis as _gtStage2Digis
# Take L1T Raw data from HLT Scouting event content
gtStage2DigisScouting = _gtStage2Digis.clone(InputLabel="hltFEDSelectorL1")
from PhysicsTools.NanoAOD.triggerObjects_cff import l1bits as _l1bits
L1BitsScoutingOnline = _l1bits.clone(src="gtStage2DigisScouting")
L1BitsSequence = cms.Sequence(gtStage2DigisScouting + L1BitsScoutingOnline)

ScoutingEBRechitAnalyzerOnline = ScoutingEBRechitAnalyzer.clone(
    L1TriggerResults = cms.InputTag('L1BitsScoutingOnline'),
    topFolderName = cms.string('HLT/ScoutingOnline/EBRechits')
)

ScoutingEBCleanedRechitAnalyzerOnline = ScoutingEBCleanedRechitAnalyzer.clone(
    L1TriggerResults = cms.InputTag('L1BitsScoutingOnline'),
    topFolderName = cms.string('HLT/ScoutingOnline/EBCleanedRechits')
)

ScoutingHBHERechitAnalyzerOnline = ScoutingHBHERechitAnalyzer.clone(
    L1TriggerResults = cms.InputTag('L1BitsScoutingOnline'),
    topFolderName = cms.string('HLT/ScoutingOnline/HBHERechits')
)

ScoutingRecHitsMonitoring = cms.Sequence(L1BitsSequence +
                                         ScoutingEBRechitAnalyzerOnline +
                                         ScoutingEBCleanedRechitAnalyzerOnline +
                                         ScoutingHBHERechitAnalyzerOnline)
