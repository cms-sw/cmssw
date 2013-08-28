import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonOfflineAnalysis")

process.load("DQMOffline.Trigger.TnPEfficiency_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# load this thing to count bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring ( 
       '/store/relval/CMSSW_3_4_0_pre4/RelValZMM/GEN-SIM-RECO/STARTUP3X_V11-v1/0000/EA1E76E3-CCC7-DE11-9A27-000423D9989E.root',
       '/store/relval/CMSSW_3_4_0_pre4/RelValJpsiMM/GEN-SIM-RECO/STARTUP3X_V11-v1/0000/C2751E35-44C8-DE11-9B85-000423D99B3E.root'
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
	 fileName = cms.untracked.string('file:/tmp/TnPEfficiency.root')
)

process.analyzerpath = cms.Path(
    process.TnPEfficiency*
    process.MEtoEDMConverter*
    process.dqmStoreStats
)

process.outpath = cms.EndPath(process.out)
