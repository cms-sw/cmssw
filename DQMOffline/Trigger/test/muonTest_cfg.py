import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing



process = cms.Process("HLTMuonOfflineAnalysis")

#process.load("DQMOffline.Trigger.MuonTrigRateAnalyzer_cosmics_cfi")
#process.load("DQMOffline.Trigger.MuonOffline_Trigger_cosmics_cff")
process.load("DQMOffline.Trigger.MuonOffline_Trigger_cff")
process.load("DQMOffline.Trigger.QuadJetAna_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")


## parse some command line arguments

options = VarParsing.VarParsing ('standard')
options.output = 'file:/data/ndpc0/b/slaunwhj/scratch0/EDM_ttbar_n2000_NewPath.root'
options.maxEvents = 10

options.parseArguments()


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
	#input = cms.untracked.int32(-1)
)


chosenFileNames = ['/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0008/2C8CD8FE-B6B5-DE11-ACB8-001D09F2905B.root',
				   '/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0007/DC830720-06B5-DE11-A22D-001D09F292D1.root',
				   '/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0007/A49B5C4C-09B5-DE11-AA7F-001D09F297EF.root',
				   '/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0007/9A88D6A4-08B5-DE11-8302-001D09F23174.root',
				   '/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0007/7EF8FC1B-0DB5-DE11-B0D3-0019B9F6C674.root',
				   '/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0007/5622E1E5-07B5-DE11-8029-001D09F2960F.root',
				   '/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0007/0E7CA853-07B5-DE11-9FAE-000423D99AA2.root',
				   '/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0007/06355619-0BB5-DE11-A99B-001D09F2924F.root']

if len(options.files) > 0:
 	print "muonTest_cfg.py === Using files from command line input"
 	chosenFileNames = options.files
else:
 	print "muonTest_cfg.py === Using default ttbar files"


process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),

							# this uses the files provided as arguments to
							# the script
							#fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0008/2C8CD8FE-B6B5-DE11-ACB8-001D09F2905B.root'),


							# These files are part of /RelValTTbar/CMSSW_3_4_0_pre1-MC_31X_V9-v1/GEN-SIM-RECO
							
							fileNames = cms.untracked.vstring(chosenFileNames)

)


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
	 outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
	 # Disable fast cloning to resolve 34X issue
	 fastCloning = cms.untracked.bool(False),						   
	 fileName = cms.untracked.string(options.output)
	 #fileName = cms.untracked.string('file:/data/ndpc0/b/slaunwhj/scratch0/EDM_zmm_fullTNP.root')
)

process.analyzerpath = cms.Path(
    process.muonFullOfflineDQM*
	process.quadJetAna *
    process.MEtoEDMConverter*
	process.dqmStoreStats
)

process.outpath = cms.EndPath(process.out)
