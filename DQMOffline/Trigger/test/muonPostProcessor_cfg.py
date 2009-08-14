import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMOffline.Trigger.MuonPostProcessor_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
									destinations = cms.untracked.vstring('cout'),
									categories = cms.untracked.vstring('DQMGenericClient'),
									debugModules = cms.untracked.vstring('*'),
									threshold = cms.untracked.string('DEBUG'),
									DQMGenericClient = cms.untracked.PSet(
	                                     #threshold = cms.untracked.string('DEBUG'),
	                                     limit = cms.untracked.int32(100000)
										 )
									)

process.source = cms.Source("PoolSource",
							# must precede with file:
							#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/MuonTrigOffline_nALL_useAodAndRAW_vMorePlots_DRStudy.root')
							#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/EDM_cosmic_vMoreTrigs_newAna.root'),
							#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/EDM_jpsi_pre10.root'),
							fileNames = cms.untracked.vstring('file:/data/ndpc0/b/slaunwhj/scratch0/EDM_test_8-14.root'),
							#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/MuonTrigOffline_nALL_p1_vMorePlots.root')
)


#process.hLTMuonPostVal.outputFileName = cms.untracked.string('/afs/cern.ch/user/s/slaunwhj/scratch0/MuonTrigPostProcessor_histos_vMoreTrigs_newAna.root')
#process.hLTMuonPostVal.outputFileName = cms.untracked.string('/afs/cern.ch/user/s/slaunwhj/scratch0/Histos_cosmic_vMoreTrigs_newAna.root')
#process.hLTMuonPostVal.outputFileName = cms.untracked.string('/afs/cern.ch/user/s/slaunwhj/scratch0/Histos_jpsi_pre10.root')
process.hLTMuonPostVal.outputFileName = cms.untracked.string('file:/data/ndpc0/b/slaunwhj/scratch0/Histos_test_8-14.root')
process.path1 = cms.Path(process.EDMtoMEConverter*process.hLTMuonPostVal*process.dqmStoreStats)
process.DQMStore.referenceFileName = ''
