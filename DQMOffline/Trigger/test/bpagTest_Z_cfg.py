import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonOfflineAnalysis")

#process.load("DQMOffline.Trigger.MuonTrigRateAnalyzer_cosmics_cfi")
process.load("DQMOffline.Trigger.BPAGTrigRateAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# load this thing to count bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring ( 
       '/store/relval/CMSSW_3_3_0_pre6/RelValZMM/GEN-SIM-RECO/STARTUP31X_V8-v1/0006/9272C1DC-1BB1-DE11-9B1E-001D09F242EF.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValZMM/GEN-SIM-RECO/STARTUP31X_V8-v1/0006/4A24952B-17B1-DE11-86F5-001D09F29597.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValZMM/GEN-SIM-RECO/STARTUP31X_V8-v1/0005/F6DCBFD7-08B1-DE11-9562-0019B9F70468.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValZMM/GEN-SIM-RECO/STARTUP31X_V8-v1/0004/E2CF654A-90AF-DE11-B770-000423D9A2AE.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValZMM/GEN-SIM-RECO/STARTUP31X_V8-v1/0004/BA76A050-94AF-DE11-9698-001D09F290CE.root',
       '/store/relval/CMSSW_3_3_0_pre6/RelValZMM/GEN-SIM-RECO/STARTUP31X_V8-v1/0004/26298EC4-91AF-DE11-84A1-001D09F28EA3.root'
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
	 fileName = cms.untracked.string('file:/tmp/Z.root')
)

process.bpagTrigOffDQM.MassParameters = cms.untracked.vdouble(100, 65, 115)
process.bpagTrigOffDQM.PtParameters = cms.untracked.vdouble(10.0, 30.0, 40., 50., 60., 100.0)
 
process.analyzerpath = cms.Path(
    process.bpagTrigOffDQM*
    process.MEtoEDMConverter*
	process.dqmStoreStats
)

process.outpath = cms.EndPath(process.out)
