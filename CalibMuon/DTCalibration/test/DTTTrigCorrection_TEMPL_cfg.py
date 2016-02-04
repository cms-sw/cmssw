import FWCore.ParameterSet.Config as cms

process = cms.Process("DTTTrigCorrection")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(RUNNUMBERTEMPLATE)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.calibDB = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPL/ttrig/ttrig_first_RUNNUMBERTEMPLATE.db'),
    authenticationMethod = cms.untracked.uint32(0)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPL/ttrig/ttrig_second_RUNNUMBERTEMPLATE.db'),
    authenticationMethod = cms.untracked.uint32(0),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    ))
)

process.DTTTrigCorrectionFirst = cms.EDAnalyzer("DTTTrigCorrectionFirst",
    debug = cms.untracked.bool(False),
    #ttrigMax = cms.untracked.double(2600.0),
    #ttrigMin = cms.untracked.double(2400.0),
    #rmsLimit = cms.untracked.double(2.)                                          
    ttrigMax = cms.untracked.double(600.0),
    ttrigMin = cms.untracked.double(200.0),
    rmsLimit = cms.untracked.double(8.)
)

process.p = cms.Path(process.DTTTrigCorrectionFirst)
