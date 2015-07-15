import FWCore.ParameterSet.Config as cms

BTVHLTOfflineClient = cms.EDAnalyzer("BTVHLTOfflineClient",

                                 processname = cms.string("HLT"),
                                 DQMDirName=cms.string("HLT/BTV"),
                                 hltTag = cms.string("HLT")

)


