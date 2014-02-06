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
        #'/store/relval/CMSSW_7_0_0_pre9/RelValZMM/GEN-SIM-RECO/START70_V2-v4/00000/CAD227AF-9E5D-E311-B8EE-0025905A608E.root',
        #'/store/relval/CMSSW_7_0_0_pre9/RelValZEE/GEN-SIM-RECO/START70_V2-v4/00000/40D06C04-965D-E311-A0A6-0025905A6090.root',
        '/store/relval/CMSSW_7_1_0_pre1/RelValZMM_13/GEN-SIM-RECO/POSTLS170_V1-v1/00000/98CCC248-5B86-E311-B20C-02163E00EB5A.root'
        ),
    secondaryFileNames = cms.untracked.vstring(
        #'/store/relval/CMSSW_7_0_0_pre9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/1800D9BB-795D-E311-BE4B-0025905A60F4.root',
        #'/store/relval/CMSSW_7_0_0_pre9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/1A9C0FD0-775D-E311-9FFC-0025905A607A.root',
        #'/store/relval/CMSSW_7_0_0_pre9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/743D862C-775D-E311-91AF-0025905A60B4.root',
        #'/store/relval/CMSSW_7_0_0_pre9/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/AE8D1115-795D-E311-96FD-0025905A611C.root',
        #'/store/relval/CMSSW_7_0_0_pre9/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/46A2C2A4-6D5D-E311-AF34-0025905A6126.root',
        #'/store/relval/CMSSW_7_0_0_pre9/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/7225FB21-775D-E311-A071-0025905A6118.root',
        #'/store/relval/CMSSW_7_0_0_pre9/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/B4DEC9EF-7A5D-E311-A90C-0025905A60A0.root',
        #'/store/relval/CMSSW_7_0_0_pre9/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2-v4/00000/C21C45BB-795D-E311-BDB1-0025905A60A6.root'
        )
)

process.DQMStore = cms.Service("DQMStore")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.destinations += ['ExoticaValidationMessages']
process.MessageLogger.categories   += ['ExoticaValidation']
#process.MessageLogger.debugModules += ['HLTExoticaValidator','HLTExoticaSubAnalysis','HLTExoticaPlotter']
process.MessageLogger.debugModules += ['*']
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
