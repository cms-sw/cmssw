import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

process.load("CondCore.CondDB.CondDB_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.INFO.limit = -1

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# output database
process.CondDB.connect = 'sqlite_file:EcalADCToGeVConstant.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB, 
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('EcalADCToGeVConstantRcd'),
            tag = cms.string('EcalADCToGeVConstant')
        )
    )
)

process.TagWriter = cms.EDAnalyzer("EcalADCToGeVTagWriter",
    record = cms.string('EcalADCToGeVConstantRcd'),
    loggingOn = cms.untracked.bool(False),
    IsDestDbCheckedInQueryLog = cms.untracked.bool(False),
    SinceAppendMode = cms.bool(True),
    Source = cms.PSet(
        since = cms.uint32(1),
        ebConst = cms.double(0.05),
        eeConst = cms.double(0.)
    )
)

process.p = cms.Path(process.TagWriter)
