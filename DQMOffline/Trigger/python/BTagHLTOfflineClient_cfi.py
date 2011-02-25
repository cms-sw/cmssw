import FWCore.ParameterSet.Config as cms

btagHLTOfflineClient = cms.EDAnalyzer("BTagHLTOfflineClient",

                                 processname = cms.string("HLT"),
                                 DQMDirName=cms.string("HLT/BTagMu"),
                                 hltTag = cms.string("HLT")

                                 #-----
                                 
)


