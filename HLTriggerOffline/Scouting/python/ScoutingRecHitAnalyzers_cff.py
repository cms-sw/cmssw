import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ScoutingEBRechitAnalyzer = DQMEDAnalyzer('ScoutingEBRecHitAnalyzer',
                                         src = cms.InputTag ("hltScoutingRecHitPacker", "EB"),
                                         L1TriggerResults = cms.InputTag('L1BitsScouting'),
                                         HLTTriggerResults = cms.InputTag('TriggerResults', '', 'HLT'),
                                         topFolderName = cms.string('HLT/ScoutingOffline/EBRechits'),
                                         lazy_eval = cms.untracked.bool(False),
                                         cut = cms.string(''),
                                         triggers = cms.VPSet(
                                             cms.PSet(
                                                 expr = cms.vstring('DST_PFScouting_JetHT'),
                                                 name = cms.string('')
                                             )
                                         ))

ScoutingEBCleanedRechitAnalyzer = ScoutingEBRechitAnalyzer.clone(
                                         src = cms.InputTag ("hltScoutingRecHitPacker", "EBCleaned"),
                                         topFolderName = cms.string('HLT/ScoutingOffline/EBCleanedRechits'))

ScoutingHBHERechitAnalyzer = DQMEDAnalyzer('ScoutingHBHERecHitAnalyzer',
                                           src = cms.InputTag("hltScoutingRecHitPacker", "HBHE"),
                                           L1TriggerResults = cms.InputTag('L1BitsScouting'),
                                           HLTTriggerResults = cms.InputTag('TriggerResults', '', 'HLT'),
                                           topFolderName = cms.string('HLT/ScoutingOffline/HBHERechits'),
                                           lazy_eval = cms.untracked.bool(False),
                                           cut = cms.string(''),
                                           triggers = cms.VPSet(
                                               cms.PSet(
                                                   expr = cms.vstring('DST_PFScouting_JetHT'),
                                                   name = cms.string('')
                                               )
                                           ))

################################################
# unpack and pack back the L1T results
################################################
from PhysicsTools.NanoAOD.triggerObjects_cff import l1bits as _l1bits
L1BitsScouting = _l1bits.clone(src="gtStage2Digis")

hltScoutingMonitoringRecHits = cms.Sequence(L1BitsScouting +
                                            ScoutingEBRechitAnalyzer +
                                            ScoutingEBCleanedRechitAnalyzer +
                                            ScoutingHBHERechitAnalyzer)
