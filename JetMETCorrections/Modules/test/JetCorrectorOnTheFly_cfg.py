import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.TFileService=cms.Service("TFileService",fileName=cms.string('JECplots.root'))
##-------------------- Communicate with the DB -----------------------
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

##-------------------- Import the JEC services -----------------------
process.load('JetMETCorrections.Configuration.JetCorrectors_cff')
##process.ak4CaloL1FastjetCorrector.useCondDB = False
#process.ak4PFL1FastjetCorrector.useCondDB = False
##-------------------- Define the source  ----------------------------
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1000)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_7_2_0_pre7/RelValProdTTbar/AODSIM/PRE_STA72_V4-v1/00000/3E58BB46-BD4B-E411-B2EC-002618943856.root')
)
process.TFileService=cms.Service("TFileService",fileName=cms.string('plots.root'))
##-------------------- User analyzer  --------------------------------
#process.testCalo  = cms.EDAnalyzer('CaloJetCorrectorOnTheFly',
    #JetCorrectionService     = cms.string('ak5CaloL1L2L3Residual'),
    #JetCollectionName        = cms.string('ak5CaloJets'),
    #MinRawJetPt              = cms.double(10),
    #Debug                    = cms.bool(True)
#)
process.testPF  = cms.EDAnalyzer('PFJetCorrectorOnTheFly',
    JetCorrector             = cms.InputTag('ak4PFL2L3ResidualCorrector'),
    JetCollectionName        = cms.InputTag('ak4PFJets'),
    MinRawJetPt              = cms.double(7),
    Debug                    = cms.bool(True)
)
#process.p = cms.Path(process.ak4CaloL2L3ResidualCorrectorChain * process.testCalo *
                      #process.ak4PFL2L3ResidualCorrectorChain * process.testPF)
process.p = cms.Path(process.ak4PFL2L3ResidualCorrectorChain * process.testPF)

