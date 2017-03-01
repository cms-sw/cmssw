import FWCore.ParameterSet.Config as cms

process = cms.Process("TTrigPreCalibProc")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.globaltag = "STARTUP_31X::All"

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.source = cms.Source("PoolSource",
    useCSA08Kludge = cms.untracked.bool(True),
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt100/ALCARECO/IDEAL_31X_StreamALCARECORpcCalHLT_v1/0007/FA1A3CD3-514F-DE11-A117-001617E30D12.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("DQMOffline.CalibMuon.dtPreCalibrationTask_cfi")
process.dtPreCalibTask.minTriggerWidth = cms.untracked.int32(400)
process.dtPreCalibTask.maxTriggerWidth = cms.untracked.int32(1200)

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
