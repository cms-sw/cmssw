import FWCore.ParameterSet.Config as cms

jetMETDQMOfflineClient = cms.EDFilter("JetMETDQMOfflineClient",

                                 Verbose=cms.untracked.int32(0),
                                 DQMDirName=cms.untracked.string("JetMET"),
                                 DQMJetDirName=cms.untracked.string("Jet"),
                                 DQMMETDirName=cms.untracked.string("MET")
                                 
)


