import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing


process = cms.Process("EDMtoMEConvert")

#process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load('Configuration/StandardSequences/EDMtoMEAtJobEnd_cff')
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMOffline.Trigger.MuonPostProcessor_cff")
#process.load("DQMOffline.Trigger.BPAGPostProcessor_cff")
#process.load("DQMOffline.Trigger.QuadJetPostProcessor_cfi")
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


options.output = 'quadjet_client_01.root'
options.maxEvents = 100
options.outputDir = '/afs/cern.ch/user/s/sboutle/scratch0/CMSSW_3_2_5/src/DQMOffline/Trigger/test/'
options.workflow  = '/DQMGeneric/BPAG/Post'
options.parseArguments()
#options.files = 'quadjet_source_01_numEvent100.root'




process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
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
							fileNames = cms.untracked.vstring(options.files),
							#fileNames = cms.untracked.vstring('file:quadjet_source_301009_numEvent100.root'),
							#fileNames = cms.untracked.vstring('file:quadjet_source_031109a_numEvent100.root'),
							#fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/s/slaunwhj/scratch0/MuonTrigOffline_nALL_p1_vMorePlots.root')
)

#three slashes, /file/descript/name
process.dqmSaver.workflow = options.workflow
# local directory to put file
process.dqmSaver.dirName = options.outputDir




process.path = cms.Path(process.EDMtoME * process.hLTMuonPostVal)
#process.path = cms.Path(process.EDMtoME*process.bPAGPostProcessor*process.dqmStoreStats)

process.endpath = cms.EndPath(process.dqmSaver)
