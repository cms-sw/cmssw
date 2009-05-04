import FWCore.ParameterSet.Config as cms

jetMETHLTOfflineClient = cms.EDFilter("JetMETHLTOfflineClient",

                                 processname = cms.string("HLT"),
                                 DQMDirName=cms.string("HLT/JetMET"),
                                 hltTag = cms.string("HLT")

                                 #-----
                                 
)


