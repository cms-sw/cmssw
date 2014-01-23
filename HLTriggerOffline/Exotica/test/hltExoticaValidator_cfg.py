import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTExoticaOfflineAnalysis")

process.load("HLTriggerOffline.Exotica.ExoticaValidation_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")


##############################################################################
##### Templates to change parameters in hltMuonValidator #####################
# process.hltMuonValidator.hltPathsToCheck = ["HLT_IsoMu3"]
# process.hltMuonValidator.genMuonCut = "abs(mother.pdgId) == 24"
# process.hltMuonValidator.recMuonCut = "isGlobalMuon && eta < 1.2"
##############################################################################

hltProcessName = "HLT"
process.hltExoticaValidator.hltProcessName = hltProcessName
#process.hltExoticaValidator.HWW.hltPathsToCheck = cms.vstring(
#		"HLT_Photon26",
		#		"HLT_Mu30_eta2p1_v",
		#		"HLT_IsoMu24_eta2p1_v",
		#"HLT_Ele27_WP80_v",
#		)

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = cms.string(autoCond['startup'])

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	    #'file:/afs/cern.ch/user/d/duarte/scratch0/step2_RAW2DIGI_RECO.root',
	    '/store/relval/CMSSW_7_0_0_pre9/RelValZMM/GEN-SIM-RECO/START70_V2-v4/00000/CAD227AF-9E5D-E311-B8EE-0025905A608E.root',
    ),
    secondaryFileNames = cms.untracked.vstring(
	    #'file:/afs/cern.ch/user/d/duarte/scratch0/H130GGgluonfusion_cfi_py_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root',
        '/store/relval/CMSSW_7_0_0_pre9/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/46A2C2A4-6D5D-E311-AF34-0025905A6126.root',
        '/store/relval/CMSSW_7_0_0_pre9/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/7225FB21-775D-E311-A071-0025905A6118.root',
        '/store/relval/CMSSW_7_0_0_pre9/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/B4DEC9EF-7A5D-E311-A90C-0025905A60A0.root',
        '/store/relval/CMSSW_7_0_0_pre9/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/C21C45BB-795D-E311-BDB1-0025905A60A6.root'
        )
)

process.DQMStore = cms.Service("DQMStore")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000
process.MessageLogger.destinations += ['ExoticaValidationMessages']
process.MessageLogger.categories   += ['ExoticaValidation']
process.MessageLogger.debugModules += ['*']#HLTExoticaValidator','HLTExoticaSubAnalysis','HLTExoticaPlotter']
process.MessageLogger.ExoticaValidationMessages = cms.untracked.PSet(
    threshold       = cms.untracked.string('DEBUG'),
    default         = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    ExoticaValidation = cms.untracked.PSet(limit = cms.untracked.int32(1000))
    )

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_MEtoEDMConverter_*_HLTExoticaOfflineAnalysis'),
    fileName = cms.untracked.string('hltExoticaValidator.root')
)


process.analyzerpath = cms.Path(
    process.hltExoticaValidator *
    process.MEtoEDMConverter
)


process.outpath = cms.EndPath(process.out)
