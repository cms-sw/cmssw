import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.TFileService=cms.Service("TFileService",fileName=cms.string('JECplots.root'))
##-------------------- Communicate with the DB -----------------------
process.load('CondCore.DBCommon.CondDBCommon_cfi')
process.CondDBCommon.connect = cms.string('sqlite_file:JEC_Spring10.db')

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
  process.CondDBCommon,
  toGet = cms.VPSet( 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_AK5Calo'), 
         label  = cms.untracked.string('JetCorrectorParametersCollection_Spring10_AK5Calo') 
      ),
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5Calo'), 
         label  = cms.untracked.string('JetCorrectorParametersCollection_Spring10DataV2_AK5Calo') 
      ),
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_AK5PF'), 
         label  = cms.untracked.string('JetCorrectorParametersCollection_Spring10_AK5PF') 
      ),
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5PF'), 
         label  = cms.untracked.string('JetCorrectorParametersCollection_Spring10DataV2_AK5PF') 
      ),
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_AK5JPT'), 
         label  = cms.untracked.string('JetCorrectorParametersCollection_Spring10_AK5JPT') 
      ),
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('JetCorrectorParametersCollection_Summer10_AK5JPT'), 
         label  = cms.untracked.string('JetCorrectorParametersCollection_Summer10_AK5JPT') 
      ),
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5JPT'), 
         label  = cms.untracked.string('JetCorrectorParametersCollection_Spring10DataV2_AK5JPT') 
      )
  )
)
##-------------------- Import the JEC services -----------------------
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')

##-------------------- Define the source  ----------------------------
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )
process.source = cms.Source("EmptySource")

##-------------------- User analyzer  --------------------------------
process.ak5calol2l3Residual  = cms.EDAnalyzer('JetCorrectorDemo',
    JetCorrectionService     = cms.string('ak5CaloL2L3Residual'),
    UncertaintyTag           = cms.string('Uncertainty'),
    UncertaintyFile          = cms.string('Spring10DataV2_Uncertainty_AK5Calo'),
    PayloadName              = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5Calo'),
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

process.ak5pfl2l3Residual = process.ak5calol2l3Residual.clone(
    JetCorrectionService = 'ak5PFL2L3Residual',
    UncertaintyFile      = 'Spring10DataV2_Uncertainty_AK5PF',
    PayloadName          = 'JetCorrectorParametersCollection_Spring10DataV2_AK5PF'
    )

process.ak5jptl2l3Residual = process.ak5calol2l3Residual.clone(
    JetCorrectionService = 'ak5JPTL2L3Residual',
    UncertaintyFile      = 'Uncertainty',
    PayloadName          = 'JetCorrectorParametersCollection_Spring10DataV2_AK5JPT'
    )

process.p = cms.Path(
          process.ak5calol2l3Residual *
          process.ak5pfl2l3Residual *
          process.ak5jptl2l3Residual 
)

