import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing



process = cms.Process("HLTMuonOfflineAnalysis")

#process.load("DQMOffline.Trigger.MuonTrigRateAnalyzer_cosmics_cfi")
#process.load("DQMOffline.Trigger.MuonOffline_Trigger_cosmics_cff")
process.load("DQMOffline.Trigger.MuonOffline_Trigger_cff")
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

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),

							fileNames = cms.untracked.vstring(options.files),
							#  --- one cosmic file to run on
							
							#fileNames = cms.untracked.vstring ( '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V11_227_Tosca090216_ReReco_FromSuperPointing_v2/0004/26F7DEA2-E81F-DE11-9686-0018F3D096E6.root'),


							#  --- one relval

							#fileNames = cms.untracked.vstring ( '/store/relval/CMSSW_3_1_0_pre7/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0004/F40BA55C-E641-DE11-858D-001D09F28E80.root'),
							#fileNames = cms.untracked.vstring ( '/store/relval/CMSSW_3_1_0_pre7/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0004/F69E8351-CE41-DE11-84E4-001D09F23944.root'),

							#---------- There is only ZMM relval with startup
							#---------- no ideal is available.
							
							#fileNames = cms.untracked.vstring ('/store/relval/CMSSW_3_1_0_pre10/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0008/6ABB6AD6-E357-DE11-8EBC-001D09F2437B.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0008/4C1E17B0-0458-DE11-A15D-001D09F26509.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0008/36A83DC8-E357-DE11-B8DC-000423D996B4.root'
							#								   ),

							#fileNames = cms.untracked.vstring ('file:/afs/cern.ch/user/s/slaunwhj/scratch0/crafT_RAW2DIGI_RECO_DQM.root'
							#								   )

							#fileNames = cms.untracked.vstring ('file:/data2/ndpc0/slaunwhj/3_1_0_pre10_RelValZMM/6ABB6AD6-E357-DE11-8EBC-001D09F2437B.root'
							#								   ),

							#fileNames = cms.untracked.vstring ('file:/data/ndpc0/b/slaunwhj/RelValTTbar/744E4482-1158-DE11-BDF0-001D09F2AF1E.root',
							#								   'file:/data/ndpc0/b/slaunwhj/RelValTTbar/AECEC48F-EC57-DE11-90D2-001D09F28F1B.root',
							#								   'file:/data/ndpc0/b/slaunwhj/RelValTTbar/AC4882A3-EE57-DE11-A925-001D09F24D8A.root',
							#								   'file:/data/ndpc0/b/slaunwhj/RelValTTbar/AA43DBAE-0458-DE11-B90D-001D09F23D1D.root',
							#								   'file:/data/ndpc0/b/slaunwhj/RelValTTbar/9CAF0AC0-ED57-DE11-85FE-000423D33970.root',
							#								   'file:/data/ndpc0/b/slaunwhj/RelValTTbar/68A9C5C9-EF57-DE11-8562-001D09F2503C.root',
							#								   'file:/data/ndpc0/b/slaunwhj/RelValTTbar/28B36B90-F057-DE11-BA80-0030487C6062.root',
							#								   'file:/data/ndpc0/b/slaunwhj/RelValTTbar/08DAD031-F257-DE11-A493-001D09F24D8A.root')

							

							#fileNames = cms.untracked.vstring ('/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/CC80B73A-CA57-DE11-BC2F-000423D99896.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/C68B7F1A-CD57-DE11-B706-00304879FA4A.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/9CA9BBC1-CD57-DE11-B62D-001D09F2424A.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/88AD5382-C657-DE11-831F-001D09F24498.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/7C7CDD0F-C457-DE11-8EEE-000423D951D4.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/4C30BDFF-B657-DE11-907A-001D09F24600.root',
							#								   '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0008/383036B6-0458-DE11-819F-001D09F29524.root'
							#								   )

							#fileNames = cms.untracked.vstring ( '/store/relval/CMSSW_3_1_0_pre10/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP_31X_v1/0001/EA6BFAD3-505A-DE11-8313-0018F3D0961A.root',
							#									'/store/relval/CMSSW_3_1_0_pre10/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP_31X_v1/0001/E4CFF5DE-6658-DE11-B9C1-0018F3D0970A.root',
							#									'/store/relval/CMSSW_3_1_0_pre10/RelValJpsiMM_Pt_0_20/GEN-SIM-RECO/STARTUP_31X_v1/0001/3ACDB108-C859-DE11-B62D-0018F3D09608.root'
							#									),




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
	 fileName = cms.untracked.string(options.output)							   
)

process.analyzerpath = cms.Path(
    process.muonFullOfflineDQM*
    process.MEtoEDMConverter*
	process.dqmStoreStats
)

process.outpath = cms.EndPath(process.out)
