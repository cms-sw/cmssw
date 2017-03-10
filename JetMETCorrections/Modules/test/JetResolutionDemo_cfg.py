import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.TFileService=cms.Service("TFileService", fileName=cms.string('JERplots.root'))

##-------------------- Communicate with the DB -----------------------
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

##-------------------- Define the source  ----------------------------
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/RunIISpring15DR74/TTJets_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/Asympt50ns_MCRUN2_74_V9A-v1/00000/0A85FC9E-7C03-E511-9510-008CFA197C38.root')
)

##-------------------- User analyzer  --------------------------------
process.demo  = cms.EDAnalyzer('JetResolutionDemo',
    jets = cms.InputTag('slimmedJets'),

    payload = cms.string('AK4PFchs'),

    resolutionsFile = cms.FileInPath('CondFormats/JetMETObjects/data/Summer15_V5_MC_JER_AK4PFchs.txt'),
    scaleFactorsFile = cms.FileInPath('CondFormats/JetMETObjects/data/Summer15_V5_MC_JER_SF_AK4PFchs.txt'),

    debug = cms.untracked.bool(True),
    useCondDB = cms.untracked.bool(False)
)

process.p = cms.Path(process.demo)

