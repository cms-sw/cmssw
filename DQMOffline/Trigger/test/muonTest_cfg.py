import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonOfflineAnalysis")

#process.load("DQMOffline.Trigger.MuonTrigRateAnalyzer_cosmics_cfi")
process.load("DQMOffline.Trigger.MuonTrigRateAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),

							#  --- one cosmic file to run on
							
							#fileNames = cms.untracked.vstring ( '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_227_Tosca090216_ReReco_FromSuperPointing_v2/0004/26F7DEA2-E81F-DE11-9686-0018F3D096E6.root'),


							#  --- one relval

							#fileNames = cms.untracked.vstring ( '/store/relval/CMSSW_3_1_0_pre7/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0004/F40BA55C-E641-DE11-858D-001D09F28E80.root'),
							#fileNames = cms.untracked.vstring ( '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0004/F69E8351-CE41-DE11-84E4-001D09F23944.root'),

							#---------- There is only ZMM relval with startup
							#---------- no ideal is available.
							
							fileNames = cms.untracked.vstring ('/store/relval/CMSSW_3_1_0_pre10/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0008/6ABB6AD6-E357-DE11-8EBC-001D09F2437B.root',
															   '/store/relval/CMSSW_3_1_0_pre10/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0008/4C1E17B0-0458-DE11-A15D-001D09F26509.root',
															   '/store/relval/CMSSW_3_1_0_pre10/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0008/36A83DC8-E357-DE11-B8DC-000423D996B4.root'
															   ),

							#fileNames = cms.untracked.vstring ('file:/data2/ndpc0/slaunwhj/3_1_0_pre10_RelValZMM/6ABB6AD6-E357-DE11-8EBC-001D09F2437B.root'
							#								   ),

							#fileNames = cms.untracked.vstring ('/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/CC80B73A-CA57-DE11-BC2F-000423D99896.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/C68B7F1A-CD57-DE11-B706-00304879FA4A.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/9CA9BBC1-CD57-DE11-B62D-001D09F2424A.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/88AD5382-C657-DE11-831F-001D09F24498.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/7C7CDD0F-C457-DE11-8EEE-000423D951D4.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/4C30BDFF-B657-DE11-907A-001D09F24600.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/383036B6-0458-DE11-819F-001D09F29524.root'
							#								   )



)

process.DQMStore = cms.Service("DQMStore")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules   = cms.untracked.vstring('*'),
    cout           = cms.untracked.PSet(
	# Be careful - this can print a lot of debug info
    #        threshold = cms.untracked.string('DEBUG')
	#        threshold = cms.untracked.string('INFO')
	threshold = cms.untracked.string('WARNING')
    ),
    categories     = cms.untracked.vstring('HLTMuonVal'),
    destinations   = cms.untracked.vstring('cout')
)

process.out = cms.OutputModule("PoolOutputModule",
	 outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
	 fileName = cms.untracked.string('/afs/cern.ch/user/s/slaunwhj/scratch0/EDM_TEST_newHist_vIndex_total.root')							   
)

process.analyzerpath = cms.Path(
    process.offlineDQMMuonTrig*
    process.MEtoEDMConverter*
	process.dqmStoreStats
)

process.outpath = cms.EndPath(process.out)
