import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.TFileService=cms.Service("TFileService",fileName=cms.string('JECplots.root'))
##-------------------- Communicate with the DB -----------------------
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

##-------------------- Import the JEC services -----------------------
process.load('JetMETCorrections.Configuration.JetCorrectors_cff')

##-------------------- Define the source  ----------------------------
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_7_2_0_pre7/RelValProdTTbar/AODSIM/PRE_STA72_V4-v1/00000/3E58BB46-BD4B-E411-B2EC-002618943856.root')
)

##-------------------- User analyzer  --------------------------------
process.ak4pfl2l3Residual  = cms.EDAnalyzer('JetCorrectorDemo',
    JetCorrector             = cms.InputTag('ak4PFL2L3ResidualCorrector'),
    UncertaintyTag           = cms.string('Uncertainty'),
    UncertaintyFile          = cms.string(''),
    PayloadName              = cms.string('AK4PF'),
    NHistoPoints             = cms.int32(10000),
    NGraphPoints             = cms.int32(500),
    EtaMin                   = cms.double(-5),
    EtaMax                   = cms.double(5),
    PtMin                    = cms.double(10),
    PtMax                    = cms.double(1000),
    #--- eta values for JEC vs pt plots ----
    VEta                     = cms.vdouble(0.0,1.0,2.0,3.0,4.0),
    #--- corrected pt values for JEC vs eta plots ----
    VPt                      = cms.vdouble(20,30,50,100,200),
    Debug                    = cms.untracked.bool(False),
    UseCondDB                = cms.untracked.bool(True)
)

process.p = cms.Path(process.ak4PFL2L3ResidualCorrectorChain * process.ak4pfl2l3Residual)

