import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTHiggsOfflineAnalysis")

#process.load("DQMOffline.RecoB.dqmAnalyzer_cff")
process.load("HLTriggerOffline.Higgs.HiggsValidation_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")


##############################################################################
##### Templates to change parameters in hltMuonValidator #####################
# process.hltMuonValidator.hltPathsToCheck = ["HLT_IsoMu3"]
# process.hltMuonValidator.genMuonCut = "abs(mother.pdgId) == 24"
# process.hltMuonValidator.recMuonCut = "isGlobalMuon && eta < 1.2"
##############################################################################

hltProcessName = "HLT"
#hltProcessName = "MYHLT"
process.hltHiggsValidator.hltProcessName = hltProcessName
#process.hltHiggsValidator.HWW.hltPathsToCheck = cms.vstring(
#		"HLT_Photon26",
        #		"HLT_Mu30_eta2p1_v",
        #		"HLT_IsoMu24_eta2p1_v",
        #"HLT_Ele27_WP80_v",
#		)

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = cms.string(autoCond['startup'])

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1) #-1
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'file:/afs/cern.ch/user/d/duarte/scratch0/step2_RAW2DIGI_RECO.root',
        #'file:/afs/cern.ch/user/s/sdonato/AFSwork/public/TTbar-GEN-SIM-RECO-new.root'
        #'file:/afs/cern.ch/user/s/sdonato/AFSwork/public/forJasper/ZnnHbb_GEN_SIM_RECO_trigger.root'
        #'/store/relval/CMSSW_9_1_0_pre1/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_90X_mcRun2_asymptotic_v5-v1/00000/0ED77E6C-BD10-E711-8002-0CC47A4D7678.root'
     #   '/store/relval/CMSSW_9_1_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_mcRun2_asymptotic_v5-v1/00000/266463C3-C310-E711-AABF-0CC47A4C8E5E.root'
        '/store/relval/CMSSW_9_1_0_pre3/RelValTTbar_13/GEN-SIM-RECO/91X_upgrade2017_design_IdealBS_v3_resub-v1/10000/58486B7A-B12F-E711-89E4-0025905B85EE.root'
        
    ),
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_9_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/91X_upgrade2017_design_IdealBS_v3_resub-v1/10000/0CFE027A-A62F-E711-BEE9-0CC47A4D7640.root',
        '/store/relval/CMSSW_9_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/91X_upgrade2017_design_IdealBS_v3_resub-v1/10000/46FB77AA-A62F-E711-859E-0025905A6138.root',
        '/store/relval/CMSSW_9_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/91X_upgrade2017_design_IdealBS_v3_resub-v1/10000/6E411A8B-A62F-E711-A432-0CC47A78A3F8.root',
        '/store/relval/CMSSW_9_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/91X_upgrade2017_design_IdealBS_v3_resub-v1/10000/BAF02D68-A62F-E711-97B2-0025905A60E0.root',
    )
)

#process.ak4PFJetsJEC.src = cms.InputTag("ak5PFJets")
#process.caloMet.alias = cms.string('met')
#process.hltHiggsValidator.Htaunu.recCaloMETLabel = cms.string('met')

process.DQMStore = cms.Service("DQMStore")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000
process.MessageLogger.destinations += ['HiggsValidationMessages']
process.MessageLogger.categories   += ['HiggsValidation']
process.MessageLogger.debugModules += ['*']#HLTHiggsValidator','HLTHiggsSubAnalysis','HLTHiggsPlotter']
process.MessageLogger.HiggsValidationMessages = cms.untracked.PSet(
    threshold       = cms.untracked.string('DEBUG'),
    default         = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    HiggsValidation = cms.untracked.PSet(limit = cms.untracked.int32(1000))
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_MEtoEDMConverter_*_HLTHiggsOfflineAnalysis'),
    fileName = cms.untracked.string('hltHiggsValidator.root')
)


process.analyzerpath = cms.Path(
    process.hltHiggsValidator *
    process.MEtoEDMConverter
)

process.outpath = cms.EndPath(process.out)

