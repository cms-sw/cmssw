import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonOfflineAnalysis")

#process.load("DQMOffline.Trigger.MuonTrigRateAnalyzer_cosmics_cfi")
process.load("DQMOffline.Trigger.BPAGTrigRateAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# load this thing to count bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring ( 
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0004/42E21769-32A2-DE11-A54F-00304867915A.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0003/A0B6EE13-CAA1-DE11-BE65-001A9281171E.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0003/9C0C4FF0-CCA1-DE11-91B6-001A92810AD4.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0003/2668C475-C9A1-DE11-A075-0018F3D096CA.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0003/1A413ED3-CAA1-DE11-8856-0018F3D09676.root',
       '/store/relval/CMSSW_3_3_0_pre3/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP31X_V7-v1/0003/04965931-CCA1-DE11-A28F-001A92971BDA.root'
    )
)

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
	 fileName = cms.untracked.string('file:/tmp/jpsi.root')
)

process.analyzerpath = cms.Path(
    process.bpagTrigOffDQM*
    process.MEtoEDMConverter*
	process.dqmStoreStats
)

process.outpath = cms.EndPath(process.out)
