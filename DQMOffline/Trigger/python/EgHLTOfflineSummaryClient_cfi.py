import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.EgHLTOffFiltersToMon_cfi import *
from DQMOffline.Trigger.EgHLTOffQTests_cfi import *


egHLTOffQTester = cms.EDAnalyzer("QualityTester",
     qtList = cms.untracked.FileInPath('DQMOffline/Trigger/data/EgHLTOffQualityTests.xml'),
     verboseQT = cms.untracked.bool(False),
     qtestOnEndJob =cms.untracked.bool(False),
     qtestOnEndRun =cms.untracked.bool(True),
     qTestOnEndLumi=cms.untracked.bool(False),
                               
 )



egHLTOffDQMSummaryClient = cms.EDAnalyzer("EgHLTOfflineSummaryClient",
                                        egHLTOffFiltersToMon,
                                        DQMDirName=cms.string("HLT/EgOffline"),
                                        hltTag = cms.string("HLT"),
                                        filterInactiveTriggers = cms.bool(True),
                                        runClientEndLumiBlock=cms.bool(False),
                                        runClientEndRun=cms.bool(True),
                                        runClientEndJob=cms.bool(False),
                                        usePathNames=cms.bool(True),
                                        egHLTSumQTests=cms.VPSet (
                                           cms.PSet(egHLTEleTrigRelEffQTests),
                                           cms.PSet(egHLTPhoTrigRelEffQTests),
                                       #   cms.PSet(egHLTEleTrigTPEffQTests),
                                       #   cms.PSet(egHLTTrigEleQTests),
                                       #   cms.PSet(egHLTTrigPhoQTests)
                                        ),
                                        egHLTEleQTestsForSumBit=cms.VPSet (
                                            cms.PSet(egHLTEleTrigRelEffQTests),
                                        ),
                                        egHLTPhoQTestsForSumBit=cms.VPSet (
                                            cms.PSet(egHLTPhoTrigRelEffQTests),
                                        ),
                                        eleHLTFilterNamesForSumBit=cms.vstring(
                                            "hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter",
                                            "hltL1NonIsoHLTNonIsoSingleElectronEt15EleIdDphiFilter",
                                            "hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter",
                                            "hltL1NonIsoHLTNonIsoSingleElectronLWEt10EleIdDphiFilter",
                                            ),
                                        phoHLTFilterNamesForSumBit=cms.vstring(
                                            "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter",
                                            "hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter",
                                            ),
)

egHLTOffCertSeq = cms.Sequence(egHLTOffQTester*egHLTOffDQMSummaryClient)
