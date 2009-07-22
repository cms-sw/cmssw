import FWCore.ParameterSet.Config as cms

process = cms.Process("TTrigPreCalibProc")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("FrontierConditions_GlobalTag_noesprefer_cff")

process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_ALL_V4::All"
#process.dtDBPrefer = cms.ESPrefer("PoolDBESSource","DTMapping")


process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
    useCSA08Kludge = cms.untracked.bool(True),
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(

       '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V4_StreamALCARECOMuAlCalIsolatedMu_step2_AlcaReco-v1/0008/001A49E8-93C3-DD11-9720-003048D15CFA.root',
       '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V4_StreamALCARECOMuAlCalIsolatedMu_step2_AlcaReco-v1/0008/004C94C2-A0C3-DD11-B949-003048D15DB6.root',
       '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V4_StreamALCARECOMuAlCalIsolatedMu_step2_AlcaReco-v1/0008/005D88AA-E0C3-DD11-BCF4-00304875ABEB.root'

    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("DQMOffline.CalibMuon.dtPreCalibrationTask_cfi")

# if read from RAW
#process.ttrigcalib.digiLabel = 'dtunpacker'
#process.load("EventFilter.DTRawToDigi.dtunpacker_cfi")

#process.DTMapping = cms.ESSource("PoolDBESSource",
#    DBParameters = cms.PSet(
#        messageLevel = cms.untracked.int32(0),
#        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#    ),
#    timetype = cms.string('runnumber'),
#    toGet = cms.VPSet(cms.PSet(
#    record = cms.string('DTT0Rcd'),
#    tag = cms.string('t0_CRAFT_V01_offline')
#    ), 
#                      cms.PSet(
#    record = cms.string('DTStatusFlagRcd'),
#    tag = cms.string('noise_CRAFT_V01_offline')
#    )),
#    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_30X_DT'),
#    #        string connect = "frontier://FrontierOn/CMS_COND_ON_18X_DT"
#    siteLocalConfig = cms.untracked.bool(False)
#)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dtPreCalibTask'),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DTPreCalibSummary = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG'),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('DTPreCalibSummary'),
    destinations = cms.untracked.vstring('cout')
)

process.p = cms.Path(process.dtPreCalibTask)
process.DQM.collectorHost = ''

# if read from RAW
#process.p = cms.Path(process.muonDTDigis*process.ttrigcalib)
