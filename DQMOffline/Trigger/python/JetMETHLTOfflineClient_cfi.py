import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

jetMETHLTOfflineClientAK4 = DQMEDHarvester("JetMETHLTOfflineClient",

                                 processname = cms.string("HLT"),
                                 DQMDirName=cms.string("HLT/JME/Jets/AK4"),
                                 hltTag = cms.string("HLT")

)

jetMETHLTOfflineClientAK8 = jetMETHLTOfflineClientAK4.clone( DQMDirName = cms.string('HLT/JME/Jets/AK8'))
jetMETHLTOfflineClientAK4Fwd = jetMETHLTOfflineClientAK4.clone( DQMDirName = cms.string('HLT/JME/Jets/AK4Fwd'))
jetMETHLTOfflineClientAK8Fwd = jetMETHLTOfflineClientAK4.clone( DQMDirName = cms.string('HLT/JME/Jets/AK8Fwd'))

jetMETHLTOfflineClient = cms.Sequence( jetMETHLTOfflineClientAK4 * jetMETHLTOfflineClientAK8 * jetMETHLTOfflineClientAK4Fwd * jetMETHLTOfflineClientAK8Fwd)
