import FWCore.ParameterSet.Config as cms

jetMETHLTOfflineClient = cms.EDAnalyzer("JetMETHLTOfflineClient",

                                 processname = cms.string("HLT"),
                                 DQMDirName=cms.string("HLT/JetMET"),
                                 hltTag = cms.string("HLT")

)


