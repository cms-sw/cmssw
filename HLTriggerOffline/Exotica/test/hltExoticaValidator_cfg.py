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
        #'/store/relval/CMSSW_7_1_0_pre1/RelValZMM_13/GEN-SIM-RECO/POSTLS170_V1-v1/00000/98CCC248-5B86-E311-B20C-02163E00EB5A.root'
        #'/store/relval/CMSSW_7_1_0_pre1/RelValADDMonoJet_d3MD3_13/GEN-SIM-RECO/POSTLS170_V1-v1/00000/2C2E143F-2386-E311-9DD8-02163E00E5B2.root'
        #'/store/relval/CMSSW_7_1_0_pre1/RelValQCD_Pt_600_800_13/GEN-SIM-RECO/POSTLS170_V1-v1/00000/E6B796C9-3686-E311-B7CF-02163E00EB6E.root'
        #'/store/relval/CMSSW_7_1_0_pre1/RelValZpTT_1500_13TeV_Tauola/GEN-SIM-RECO/POSTLS170_V1-v1/00000/0C7325DD-4986-E311-A342-02163E008F41.root'
        '/store/relval/CMSSW_7_1_0_pre1/RelValTTbarLepton_13/GEN-SIM-RECO/POSTLS170_V1-v1/00000/6E99FADF-4386-E311-8AB5-02163E008CCC.root',
        '/store/relval/CMSSW_7_1_0_pre1/RelValTTbarLepton_13/GEN-SIM-RECO/POSTLS170_V1-v1/00000/82E02E9D-3886-E311-A699-003048D2C0F2.root',
        '/store/relval/CMSSW_7_1_0_pre1/RelValTTbarLepton_13/GEN-SIM-RECO/POSTLS170_V1-v1/00000/5CB7C5E7-4B86-E311-9394-02163E00EB64.root')
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

process.MessageLogger.categories.extend(["GetManyWithoutRegistration","GetByLabelWithoutRegistration"])

_messageSettings = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1),
    optionalPSet = cms.untracked.bool(True),
    limit = cms.untracked.int32(10000000)
    )

process.MessageLogger.cerr.GetManyWithoutRegistration = _messageSettings
process.MessageLogger.cerr.GetByLabelWithoutRegistration = _messageSettings

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_MEtoEDMConverter_*_HLTExoticaOfflineAnalysis'),
    fileName = cms.untracked.string('hltExoticaValidator.root')
)


process.analyzerpath = cms.Path(
    process.ExoticaValidationSequence +
    process.MEtoEDMConverter
)


process.outpath = cms.EndPath(process.out)
