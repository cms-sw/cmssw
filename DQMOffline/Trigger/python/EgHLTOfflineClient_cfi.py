import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.EgHLTOffFiltersToMon_cfi import *

egHLTOffDQMClient = cms.EDFilter("EgHLTOfflineClient",
                                 egHLTOffFiltersToMon,
                                 sourceModuleName = cms.string('egammaHLTDQM'),
                                 DQMDirName=cms.string("HLT/EgammaHLTOffline_egammaHLTDQM"),

                                 #efficiences variables and vs variables
                                 eleN1EffVars=cms.vstring('dEtaIn','dPhiIn','hOverE','sigmaIEtaIEta',
                                                          'isolEm','isolHad','isolPtTrks','hltIsolHad',
                                                          'hltIsoltrksEle'),
                                 eleSingleEffVars=cms.vstring('dEtaIn','dPhiIn','hOverE','sigmaIEtaIEta',
                                                              'isolEm','isolHad','isolPtTrks',
                                                              'hltIsolHad','hltIsoltrksEle','hltIsolTrksPho'),
                                 eleEffTags=cms.vstring('effVsEt','effVsEta','effVsPhi'), #used for vs vars and T&P, Fake rate tags
                                 eleTrigTPEffVsVars=cms.vstring('et','eta','phi','charge'),
                                 eleLooseTightTrigEffVsVars=cms.vstring('et','eta','phi','charge',
                                                                        'hOverE','sigmaIEtaIEta','dPhiIn','dEtaIn'),
                                 
                                 phoN1EffVars=cms.vstring('hOverE','sigmaIEtaIEta','r9',
                                                          'isolEm','isolHad','isolPtTrks','isolNrTrks'),
                                 phoSingleEffVars=cms.vstring('sigmaIEtaIEta','hltIsolHad','hltIsolTrksPho'),
                                 phoEffTags=cms.vstring('effVsEt','effVsEta','effVsPhi'),
                                 phoTrigTPEffVsVars=cms.vstring('et','eta','phi','charge'),
                                 phoLooseTightTrigEffVsVars=cms.vstring('et','eta','phi',
                                                                        'hOverE','sigmaIEtaIEta')
                                 )


