import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMOffline.Trigger.MuonPostProcessor_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
									destinations = cms.untracked.vstring('MuonTrigPostTest.log'),
									categories = cms.untracked.vstring('MuonTrigPostLog'),
									debugModules = cms.untracked.vstring('*'),
									MuonTrigPostLog = cms.untracked.PSet(
	                                threshold = cms.untracked.string('DEBUG'),
									  MuonTrigPostLog = cms.untracked.PSet(
	                                   limit = cms.untracked.int32(100000))
									  ),																	 									
									)

process.source = cms.Source("PoolSource",
							# must precede with file:
							#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/MuonTrigOffline_nALL_useAodAndRAW_vMorePlots_DRStudy.root')
							fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/MuonTrigOffline_n10_vMorePlots.root')
							#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/MuonTrigOffline_nALL_p1_vMorePlots.root')
)

process.HLTMuonPostVal.outputFileName = cms.untracked.string('/afs/cern.ch/user/s/slaunwhj/scratch0/MuonTrigPostProcessor_histos_n10_vMorePlots.root')

process.path1 = cms.Path(process.EDMtoMEConverter*process.HLTMuonPostVal)
process.DQMStore.referenceFileName = ''
