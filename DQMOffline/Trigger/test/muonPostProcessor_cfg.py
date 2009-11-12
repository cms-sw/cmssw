import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing


process = cms.Process("EDMtoMEConvert")

#process.load("DQMServices.Components.EDMtoMEConverter_cff")
#process.load('Configuration/StandardSequences/EDMtoMEAtJobEnd_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMOffline.Trigger.MuonPostProcessor_cfi")
process.load("DQMOffline.Trigger.MuonHLTValidation_cfi")
#process.load("DQMOffline.Trigger.BPAGPostProcessor_cff")
process.load("DQMOffline.Trigger.TnPEfficiencyPostProcessor_cff")
process.load("DQMServices.Components.DQMStoreStats_cfi")

## parse some command line arguments

options = VarParsing.VarParsing ('standard')

options.register ('outputDir',
                  -1, # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Directory for DQM saver to put your output")

options.register ('workflow',
                  -1, # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "One string w/ 3 slashes: /Blah/fu/mEh")


options.output = 'file:/data/ndpc0/b/slaunwhj/scratch0/EDM_ttbar_n2000_NewPath.root'
options.maxEvents = 10
options.outputDir = '/data/ndpc1/b/slaunwhj/scratch0/'
options.workflow  = '/DQMGeneric/BPAG/Post'
options.parseArguments()





process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

process.MessageLogger = cms.Service("MessageLogger",
									destinations = cms.untracked.vstring('cout'),
									#categories = cms.untracked.vstring('DQMGenericClient'),
									categories = cms.untracked.vstring('HLTMuonVal'),
									debugModules = cms.untracked.vstring('*'),
									threshold = cms.untracked.string('INFO'),
									HLTMuonVal = cms.untracked.PSet(
	                                     #threshold = cms.untracked.string('DEBUG'),
	                                     limit = cms.untracked.int32(100000)
										 )
									)

process.source = cms.Source("PoolSource",
							# must precede with file:
							#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/MuonTrigOffline_nALL_useAodAndRAW_vMorePlots_DRStudy.root')
							#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/EDM_cosmic_vMoreTrigs_newAna.root'),
							#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/EDM_jpsi_pre10.root'),
							fileNames = cms.untracked.vstring(options.files),
							#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/MuonTrigOffline_nALL_p1_vMorePlots.root')
)


process.dqmSaver.workflow = options.workflow
process.dqmSaver.dirName = options.outputDir
#process.DQMStore.verbose = 10



process.path = cms.Path(process.EDMtoME*process.hLTMuonPostVal*process.tagAndProbeEfficiencyPostProcessor*process.muonHLTCertSeq*process.dqmStoreStats)
#process.path = cms.Path(process.EDMtoME*process.bPAGPostProcessor*process.dqmStoreStats)

process.endpath = cms.EndPath(process.dqmSaver)



