import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.TFileService=cms.Service("TFileService",fileName=cms.string('JECplots.root'))
##-------------------- Communicate with the DB -----------------------
process.load('Configuration.StandardSequences.Services_cff')
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'START38_V13::All'


##-------------------- Import the JEC services -----------------------
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')


from CondCore.DBCommon.CondDBSetup_cfi import *
process.jec = cms.ESSource("PoolDBESSource",CondDBSetup,
                   connect = cms.string("sqlite:Jec11_V10.db"),
                   toGet =  cms.VPSet(
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec11_V10_AK5Calo"),
                                label=cms.untracked.string("AK5Calo")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec11_V10_AK5PF"),
                                label=cms.untracked.string("AK5PF")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec11_V10_AK5PFchs"),
                                label=cms.untracked.string("AK5PFchs")),
                       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                tag = cms.string("JetCorrectorParametersCollection_Jec11_V10_AK5JPT"),
                                label=cms.untracked.string("AK5JPT"))
                       )
                   
                   )
es_prefer_jec = cms.ESPrefer("PoolDBESSource","jec")






##-------------------- Define the source  ----------------------------
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )
process.source = cms.Source("EmptySource")

##-------------------- User analyzer  --------------------------------
process.ak5calol2l3Residuall5l7  = cms.EDAnalyzer('FactorizedJetCorrectorDemo',
    levels                   = cms.vstring( 'L2Relative', 'L3Absolute', 'L5Flavor_gJ', 'L7Parton_gJ'),
    UncertaintyTag           = cms.string('Uncertainty'),
    UncertaintyFile          = cms.string(''),
    PayloadName              = cms.string('AK5Calo'),
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
    Debug                    = cms.untracked.bool(True)
)

process.p = cms.Path(process.ak5calol2l3Residuall5l7)

