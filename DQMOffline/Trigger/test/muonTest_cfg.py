import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonOfflineAnalysis")

#process.load("DQMOffline.Trigger.MuonTrigRateAnalyzer_cosmics_cfi")
process.load("DQMOffline.Trigger.MuonTrigRateAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

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

							fileNames = cms.untracked.vstring ('/store/relval/CMSSW_3_1_0_pre9/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0007/90691004-D050-DE11-BC8D-000423D9939C.root',
															   '/store/relval/CMSSW_3_1_0_pre9/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0007/426B56FE-9A50-DE11-87C5-0019B9F72D71.root',
															   '/store/relval/CMSSW_3_1_0_pre9/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0007/04788C19-9950-DE11-A698-001D09F24024.root',
															   '/store/relval/CMSSW_3_1_0_pre9/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0006/A2B552C3-BD4E-DE11-9A64-001D09F251BD.root',
															   '/store/relval/CMSSW_3_1_0_pre9/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0006/025E9766-BE4E-DE11-BB1C-001D09F2A465.root'),


)

process.DQMStore = cms.Service("DQMStore")

process.MessageLogger = cms.Service("MessageLogger",
   HLTMuonVallog  = cms.untracked.PSet(
        threshold  = cms.untracked.string('INFO'),
        default    = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        HLTMuonVal = cms.untracked.PSet(
            limit = cms.untracked.int32(100000)
        )
    ),
    debugModules   = cms.untracked.vstring('*'),
    cout           = cms.untracked.PSet(
	# Be careful - this can print a lot of debug info
    #        threshold = cms.untracked.string('DEBUG')
	        threshold = cms.untracked.string('WARNING')
    ),
    categories     = cms.untracked.vstring('HLTMuonVal'),
    destinations   = cms.untracked.vstring('cout', 'HLTMuonVal.log')
)

process.out = cms.OutputModule("PoolOutputModule",
	 outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
	 fileName = cms.untracked.string('/afs/cern.ch/user/s/slaunwhj/scratch0/EDM_cosmics_n10_vNewConfig.root')
)

process.analyzerpath = cms.Path(
    process.offlineDQMMuonTrig*
    process.MEtoEDMConverter
)

process.outpath = cms.EndPath(process.out)
