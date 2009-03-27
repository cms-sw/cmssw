import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.EgHLTOffFiltersToMon_cfi import *
from DQMOffline.Trigger.EgHLTOffQTests_cfi import *

egHLTOffDQMSummaryClient = cms.EDFilter("EgHLTOfflineSummaryClient",
                                        egHLTOffFiltersToMon,
                                        sourceModuleName = cms.string('egammaHLTDQM'),
                                        DQMDirName=cms.string("HLT/EgammaHLTOffline_egammaHLTDQM"),
                                        egHLTSumQTests=cms.VPSet (
                                            cms.PSet(egHLTEleTrigRelEffQTests),
                                            cms.PSet(egHLTPhoTrigRelEffQTests),
                                            cms.PSet(egHLTEleTrigTPEffQTests),
                                            cms.PSet(egHLTTrigEleQTests),
                                            cms.PSet(egHLTTrigPhoQTests)
                                            )
                         #               egHLTSumQTests=cms.vstring('Ele Rel Trig Eff:*trigEffTo*gsfEle_trigCuts*',
                          #                                         'Pho Rel Trig Eff:*trigEffTo*pho_trigCuts*',
                           #                                        'Ele T&P Trig Eff:*trigTagProbeEff_gsfEle*',
                            #                                       'Triggered Ele:*gsfEle_effVs*_n1*',
                             #                                      'Triggered Pho:*pho_effVs*_n1*',
                              #                                     )
                                        )
