import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.EgHLTOffFiltersToMon_cfi import *

egHLTOffDQMClient = cms.EDAnalyzer("EgHLTOfflineClient",
                                 egHLTOffFiltersToMon,
                                 DQMDirName=cms.string("HLT/EgOffline"),
                                 hltTag = cms.string("HLT"),
                                 filterInactiveTriggers = cms.bool(True),
                                 runClientEndLumiBlock=cms.bool(False),
                                 runClientEndRun=cms.bool(True),
                                 runClientEndJob=cms.bool(False),

                                 #efficiences variables and vs variables
                                 #-----Morse-----
                                 #eleN1EffVars=cms.vstring('dEtaIn','dPhiIn','hOverE','sigmaIEtaIEta',
                                 #                         'isolEm','isolHad','isolPtTrks','hltIsolHad',
                                 #                         'hltIsoltrksEle'),
                                 eleN1EffVars=cms.vstring('dEtaIn','dPhiIn','hOverE','sigmaIEtaIEta','maxr9',
                                                          'isolEm','isolHad','isolPtTrks','hltIsolHad',
                                                          'hltIsoltrksEle'),
                                 #-------------------
                                 eleSingleEffVars=cms.vstring('dEtaIn','dPhiIn','hOverE','sigmaIEtaIEta',
                                                              'isolEm','isolHad','isolPtTrks','hltIsolHad',
                                                              'hltIsoltrksEle','hltIsolTrksPho'),
                                 eleEffTags=cms.vstring('effVsEt','effVsEta','effVsPhi'), #used for vs vars and T&P, Fake rate tags
                                 eleTrigTPEffVsVars=cms.vstring('et','eta','phi','charge','etUnCut'),
                                 eleLooseTightTrigEffVsVars=cms.vstring('et','eta','phi','charge','etUnCut',
                                                                        'hOverE','sigmaIEtaIEta','dPhiIn','dEtaIn'),
                                 #----Morse-----
                                 #phoN1EffVars=cms.vstring('hOverE','sigmaIEtaIEta','r9',
                                 #                         'isolEm','isolHad','isolPtTrks','isolNrTrks'),
                                 phoN1EffVars=cms.vstring('hOverE','sigmaIEtaIEta','maxr9',
                                                          'isolEm','isolHad','isolPtTrks','isolNrTrks'),
                                 #-------------
                                 phoSingleEffVars=cms.vstring('sigmaIEtaIEta','isolEm','isolHad',
                                                              'hltIsolHad','hltIsolTrksPho'),
                                 phoEffTags=cms.vstring('effVsEt','effVsEta','effVsPhi'),
                                 phoTrigTPEffVsVars=cms.vstring('et','eta','phi','charge','etUnCut'),
                                 phoLooseTightTrigEffVsVars=cms.vstring('et','eta','phi','etUnCut',
                                                                        'hOverE','sigmaIEtaIEta')
                                 )


