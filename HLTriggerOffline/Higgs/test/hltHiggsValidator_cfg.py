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
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'file:/afs/cern.ch/user/d/duarte/scratch0/step2_RAW2DIGI_RECO.root',
        'file:/afs/cern.ch/user/s/sdonato/AFSwork/public/TTbar-GEN-SIM-RECO-new.root'
        #'file:/afs/cern.ch/user/s/sdonato/AFSwork/public/forJasper/ZnnHbb_GEN_SIM_RECO_trigger.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
        #'file:/afs/cern.ch/user/d/duarte/scratch0/H130GGgluonfusion_cfi_py_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root',
        #'file:/afs/cern.ch/work/j/jlauwers/hlt/CMSSW_7_2_0_pre1/src/GEN-SIM-RAW/0090828A-AC71-E311-A488-7845C4FC36D7.root'
    )
)

#process.ak4PFJetsJEC.src = cms.InputTag("ak5PFJets")
#process.caloMet.alias = cms.string('met')
#process.hltHiggsValidator.Htaunu.recCaloMETLabel = cms.string('met')

process.DQMStore = cms.Service("DQMStore")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000

process.MessageLogger.debugModules += ['*']#HLTHiggsValidator','HLTHiggsSubAnalysis','HLTHiggsPlotter']
process.MessageLogger.files.HiggsValidationMessages = cms.untracked.PSet(
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
