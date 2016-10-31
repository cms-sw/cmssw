import FWCore.ParameterSet.Config as cms

jetMETHLTOfflineClientAK4 = cms.EDAnalyzer("JetMETHLTOfflineClient",

                                 processname = cms.string("HLT"),
                                 DQMDirName=cms.string("HLT/JetMET"),
                                 hltTag = cms.string("HLT")

)

jetMETHLTOfflineClientAK8 = jetMETHLTOfflineClientAK4.clone( DQMDirName = cms.string('HLT/JetMET/AK8'))

jetMETHLTOfflineClient = cms.Sequence( jetMETHLTOfflineClientAK4 * jetMETHLTOfflineClientAK8 )
