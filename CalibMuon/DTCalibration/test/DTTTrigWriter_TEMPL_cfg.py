import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("CondCore.DBCommon.CondDBSetup_cfi")

#process.source = cms.Source("EmptySource",
#    numberEventsInRun = cms.untracked.uint32(1),
#    firstRun = cms.untracked.uint32(RUNNUMBERTEMPLATE)
#)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    DBParameters = cms.PSet(),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPL/ttrig/ttrig_first_RUNNUMBERTEMPLATE.db'),
    authenticationMethod = cms.untracked.uint32(0),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    ))
)

process.load("CalibMuon.DTCalibration.DTTTrigWriter_cfi")
process.ttrigwriter.kFactor = -0.7
process.ttrigwriter.rootFileName = "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPL/ttrig/DTTimeBoxes_RUNNUMBERTEMPLATE.root"

process.p = cms.Path(process.ttrigwriter)
