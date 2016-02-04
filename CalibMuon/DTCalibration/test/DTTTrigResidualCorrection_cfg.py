import FWCore.ParameterSet.Config as cms

process = cms.Process("DTTTrigCorrection")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('*')
process.MessageLogger.destinations = cms.untracked.vstring('cerr')
process.MessageLogger.categories.append('Calibration')
process.MessageLogger.cerr =  cms.untracked.PSet(
     threshold = cms.untracked.string('DEBUG'),
     noLineBreaks = cms.untracked.bool(False),
     DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
     INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
     Calibration = cms.untracked.PSet(limit = cms.untracked.int32(-1))
)

#process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRAFT_ALL_V4::All"
 
process.load("CalibMuon.DTCalibration.dt_DBLocal_cff")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.ttrig = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/test/ttrig/ttrig_matchRPhi_67234.db'),
    authenticationMethod = cms.untracked.uint32(0)
)

process.es_prefer_calibDB = cms.ESPrefer('PoolDBESSource','ttrig')

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:ttrig_ResidualCorr_67234.db'),
    authenticationMethod = cms.untracked.uint32(0),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    ))
)

process.DTTTrigCorrection = cms.EDAnalyzer("DTTTrigCorrection",
    correctionAlgo = cms.string('DTTTrigResidualCorrection'),
    correctionAlgoConfig = cms.PSet(
        residualsRootFile = cms.string('/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/test/ttrig/DTkFactValidation_ttrig_matchRPhi_67234.root'),
        useFitToResiduals = cms.bool(True)
    )
)

process.p = cms.Path(process.DTTTrigCorrection)


