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
                                 eleN1EffVars=cms.vstring('dEtaIn','dPhiIn','hOverE','sigmaIEtaIEta',
                                                          'hltIsolEm','hltIsolHad','hltIsoltrksEle'),
                                 eleHLTvOfflineVars=cms.vstring('eta','phi','energy'),
                                 #-------------------
                                 eleSingleEffVars=cms.vstring('dEtaIn','dPhiIn','hOverE','sigmaIEtaIEta',
                                                              'hltIsolEm','hltIsolHad','hltIsoltrksEle'),
                                 eleEffTags=cms.vstring('effVsEt','effVsEta'),#'effVsPhi'), #used for vs vars and T&P, Fake rate tags
                                 eleTrigTPEffVsVars=cms.vstring('et','eta','nVertex'),#'phi','charge','etUnCut'),
                                 eleLooseTightTrigEffVsVars=cms.vstring('et','eta','hOverE','sigmaIEtaIEta','dPhiIn','dEtaIn'),
                                 #----Morse-----
                                 #phoN1EffVars=cms.vstring('hOverE','sigmaIEtaIEta','r9',
                                 #                         'isolEm','isolHad','isolPtTrks','isolNrTrks'),
                                 phoN1EffVars=cms.vstring('hOverE','sigmaIEtaIEta',
                                                          'isolEm','isolHad','isolPtTrks','isolNrTrks'),
                                 phoHLTvOfflineVars=cms.vstring('eta','phi','energy'),
                                 #-------------
                                 phoSingleEffVars=cms.vstring('hOverE','sigmaIEtaIEta','isolEm','isolHad','isolPtTracks'),
                                 phoEffTags=cms.vstring('effVsEt','effVsEta'),#'effVsPhi'),
                                 phoTrigTPEffVsVars=cms.vstring('et','eta'),#'phi','charge','etUnCut'),
                                 phoLooseTightTrigEffVsVars=cms.vstring('et','eta','hOverE','sigmaIEtaIEta')
                                 )


