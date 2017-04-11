import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.TFileService=cms.Service("TFileService", fileName=cms.string('JERplots.root'))

##-------------------- Communicate with the DB -----------------------
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

##-------------------- Define the source  ----------------------------
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10)
)

from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpMINIAODSIM

process.source = cms.Source("PoolSource",
    fileNames = filesRelValTTbarPileUpMINIAODSIM
)

##-------------------- User analyzer  --------------------------------
process.demo  = cms.EDAnalyzer('JetResolutionDemo',
    jets = cms.InputTag('slimmedJets'),

    payload = cms.string('AK4PFchs'),

    resolutionsFile = cms.FileInPath('CondFormats/JetMETObjects/data/Summer15_V0_MC_JER_AK4PFchs.txt'),
    scaleFactorsFile = cms.FileInPath('CondFormats/JetMETObjects/data/Summer12_V1_MC_JER_SF_AK5PFchs.txt'),

    debug = cms.untracked.bool(False),
    useCondDB = cms.untracked.bool(False)
)

process.p = cms.Path(process.demo)

