import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing



process = cms.Process("QuadJetAnalysis")

#process.load("DQMOffline.Trigger.MuonTrigRateAnalyzer_cosmics_cfi")
#process.load("DQMOffline.Trigger.MuonOffline_Trigger_cosmics_cff")
#process.load("DQMOffline.Trigger.MuonOffline_Trigger_cff")
process.load("DQMOffline.Trigger.QuadJetAna_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")


## parse some command line arguments

options = VarParsing.VarParsing ('standard')

#options.files = '/store/relval/CMSSW_3_2_5/RelValTTbar/GEN-SIM-RECO/MC_31X_V5-v1/0011/B46D442F-478E-DE11-BD8D-001D09F2462D.root'
options.output = 'quadjet_source_031109a.root'
options.maxEvents = 100

options.parseArguments()


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
    #   input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
	fileNames = cms.untracked.vstring(options.files),
)
#process.source.fileNames=["/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V2-v1/0002/DEFAFFEA-596B-DE11-9CC0-001D09F252F3.root"]



# take this out and try to run
process.DQMStore = cms.Service("DQMStore")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules   = cms.untracked.vstring('*'),
    cout           = cms.untracked.PSet(
	# Be careful - this can print a lot of debug info
    #        threshold = cms.untracked.string('DEBUG')
	threshold = cms.untracked.string('INFO')
	#threshold = cms.untracked.string('WARNING')
    ),
    categories     = cms.untracked.vstring('HLTMuonVal'),
    destinations   = cms.untracked.vstring('cout')
)

process.out = cms.OutputModule("PoolOutputModule",
							   fastCloning = cms.untracked.bool(False), 
	 outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
	 fileName = cms.untracked.string(options.output)							   
)

process.analyzerpath = cms.Path(
    process.quadJetAna*
    process.MEtoEDMConverter
	#process.dqmStoreStats
)

process.outpath = cms.EndPath(process.out)
