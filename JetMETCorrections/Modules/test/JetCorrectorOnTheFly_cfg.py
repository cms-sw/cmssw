import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.TFileService=cms.Service("TFileService",fileName=cms.string('JECplots.root'))
##-------------------- Communicate with the DB -----------------------
#process.load('Configuration.StandardSequences.Services_cff')
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'START38_V14::All'

##-------------------- Import the JEC services -----------------------
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
process.ak5PFL2Relative.useCondDB = False
process.ak5PFL3Absolute.useCondDB = False
process.ak5PFResidual.useCondDB = False
##-------------------- Define the source  ----------------------------
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/data/kkousour/EE8F5781-CCEB-DF11-8D07-003048F0E1E8.root')
)
process.TFileService=cms.Service("TFileService",fileName=cms.string('plots.root'))
##-------------------- User analyzer  --------------------------------
process.test  = cms.EDAnalyzer('PFJetCorrectorOnTheFly',
    JetCorrectionService     = cms.string('ak5PFL1L2L3Residual'),
    JetCollectionName        = cms.string('ak5PFJets'),
    Debug                    = cms.bool(True)
)
process.p = cms.Path(process.test)

