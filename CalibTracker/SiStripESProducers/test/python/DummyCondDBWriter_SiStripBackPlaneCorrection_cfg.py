# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("Builder")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('siStripBackPlaneCorrectionDummyDBWriter'),
    files = cms.untracked.PSet(
        BackPlaneCorrectionBuilder = cms.untracked.PSet(

        )
    ),
    threshold = cms.untracked.string('DEBUG')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibTracker.SiStripESProducers.fake.SiStripBackPlaneCorrectionFakeESSource_cfi")
process.load("CalibTracker.SiStripESProducers.DBWriter.SiStripBackPlaneCorrectionDummyDBWriter_cfi")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    connect = cms.string('sqlite_file:dbfile.db'),
    timetype = cms.untracked.string('runnumber'),    
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBackPlaneCorrectionRcd'),
        tag = cms.string('SiStripBackPlaneCorrection_peak_31X')
    ))
)

process.siStripBackPlaneCorrectionDummyDBWriter.record=process.PoolDBOutputService.toPut[0].record

from CalibTracker.SiStripESProducers.fake.SiStripBackPlaneCorrectionFakeESSource_cfi import siStripBackPlaneCorrectionFakeESSource
#BackPlaneCorrection values for each module geometry: IB1, IB2, OB1, OB2, W1A, W1B, W2A, W2B, W3A, W3B, W4, W5, W6, W7
#siStripBackPlaneCorrectionFakeESSource.BackPlaneCorrection_PerModuleGeometry = cms.vdouble(0.034, 0.034, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
siStripBackPlaneCorrectionFakeESSource.BackPlaneCorrection_PerModuleGeometry = cms.vdouble(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

process.p1 = cms.Path(process.siStripBackPlaneCorrectionDummyDBWriter)
