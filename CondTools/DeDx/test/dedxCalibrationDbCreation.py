import FWCore.ParameterSet.Config as cms

tagger = "dedxCalibration"
db_file = f'{tagger}.db'

process = cms.Process("DeDxCalibCreator")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:' + db_file

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(1),
)

process.source = cms.Source("EmptySource")
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string("DeDxCalibrationRcd"),
            tag = cms.string(tagger),
            label = cms.string(""),
        ),
    )
)

process.dbCreator = cms.EDAnalyzer("DeDxCalibrationDbCreator",
    propFile = cms.string("stripProps.par"),
    gainFile = cms.string("gain.dat")
)

process.p = cms.Path(
    process.dbCreator
)
