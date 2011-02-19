import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.TFileService=cms.Service("TFileService",fileName=cms.string('JECplots.root'))
##-------------------- Communicate with the DB -----------------------
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START38_V13::All'

##-------------------- Import the JEC services -----------------------
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')

##-------------------- Define the source  ----------------------------
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )
process.source = cms.Source("EmptySource")

##-------------------- User analyzer  --------------------------------
process.ak5pfl2l3Residual  = cms.EDAnalyzer('JetCorrectorDemo',
    JetCorrectionService     = cms.string('ak5PFL2L3Residual'),
    UncertaintyTag           = cms.string('Uncertainty'),
    UncertaintyFile          = cms.string(''),
    PayloadName              = cms.string('AK5PF'),
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

process.p = cms.Path(process.ak5pfl2l3Residual)

