import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
##-------------------- Communicate with the DB -----------------------
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START38_V14::All'

##-------------------- Import the JEC services -----------------------
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
process.ak5CaloL1Offset.useCondDB = False
process.ak5PFL1Offset.useCondDB = False
process.ak5JPTL1Offset.useCondDB = False
process.ak5L1JPTOffset.offsetService = 'ak5CaloL1Offset'
##-------------------- Define the source  ----------------------------
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/data/kkousour/EE8F5781-CCEB-DF11-8D07-003048F0E1E8.root')
)
process.TFileService=cms.Service("TFileService",fileName=cms.string('plots.root'))
##-------------------- User analyzer  --------------------------------
process.testCalo  = cms.EDAnalyzer('CaloJetCorrectorOnTheFly',
    JetCorrectionService     = cms.string('ak5CaloL1Offset'),
    JetCollectionName        = cms.string('ak5CaloJets'),
    MinRawJetPt              = cms.double(10),
    Debug                    = cms.bool(False)
)
process.testPF  = cms.EDAnalyzer('PFJetCorrectorOnTheFly',
    JetCorrectionService     = cms.string('ak5PFL1L2L3Residual'),
    JetCollectionName        = cms.string('ak5PFJets'),
    MinRawJetPt              = cms.double(7),
    Debug                    = cms.bool(False)
)
process.testJPT  = cms.EDAnalyzer('JPTJetCorrectorOnTheFly',
    JetCorrectionService     = cms.string('ak5L1JPTOffset'),
    JetCollectionName        = cms.string('JetPlusTrackZSPCorJetAntiKt5'),
    MinRawJetPt              = cms.double(15),
    Debug                    = cms.bool(True)
)
process.p = cms.Path(process.testCalo * process.testPF * process.testJPT)

