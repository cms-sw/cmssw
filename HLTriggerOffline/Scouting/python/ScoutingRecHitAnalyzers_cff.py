import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ScoutingEBRechitAnalyzer = DQMEDAnalyzer('ScoutingEBRecHitAnalyzer',
                                         src = cms.InputTag ("hltScoutingRecHitPacker", "EB"),
                                         L1TriggerResults = cms.InputTag('*'),
                                         HLTTriggerResults = cms.InputTag('TriggerResults', '', 'HLT'),
                                         lazy_eval = cms.untracked.bool(False),
                                         cut = cms.string(''),
                                         triggers = cms.VPSet(
                                             cms.PSet(
                                                 expr = cms.vstring('DST_PFScouting_JetHT'),
                                                 name = cms.string('')
                                             )                                             
                                         ))

ScoutingHBHERechitAnalyzer = DQMEDAnalyzer('ScoutingHBHERecHitAnalyzer',
                                         src = cms.InputTag("hltScoutingRecHitPacker", "HBHE"),
                                         L1TriggerResults = cms.InputTag('*'),
                                         HLTTriggerResults = cms.InputTag('TriggerResults', '', 'HLT'),
                                         lazy_eval = cms.untracked.bool(False),
                                         cut = cms.string(''),
                                         triggers = cms.VPSet(
                                             cms.PSet(
                                                 expr = cms.vstring('DST_PFScouting_JetHT'),
                                                 name = cms.string('')
                                             )                                             
                                         ))

hltScoutingMonitoringRecHits = cms.Sequence(ScoutingEBRechitAnalyzer + ScoutingHBHERechitAnalyzer)
