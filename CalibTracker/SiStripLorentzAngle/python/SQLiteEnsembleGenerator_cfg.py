#based on CalibTracker/SiStripESProducers/test/python/DummyCondDBWriter_SiStripLorentzAngle_cfg.py

import FWCore.ParameterSet.Config as cms

process = cms.Process("Builder")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('siStripLorentzAngleDummyDBWriter'),
    files = cms.untracked.PSet(
        SQLiteGenerator = cms.untracked.PSet(

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

process.load("CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi")
process.load("CalibTracker.SiStripESProducers.DBWriter.SiStripLorentzAngleDummyDBWriter_cfi")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    connect = cms.string('sqlite_file:SiStripLorentzAngle_CalibrationEnsemble.db'),
    timetype = cms.untracked.string('runnumber'),    
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripLorentzAngleRcd'),
        tag = cms.string('SiStripLorentzAngle_CalibrationEnsemble_31X')
    ))
)

process.siStripLorentzAngleDummyDBWriter.record=process.PoolDBOutputService.toPut[0].record

# Three possible generations:
# - give two values = (min,max)                                                -> uniform distribution
# - give one value and PerCent_Err != 0                                        -> gaussian distribution
# - either give two equal values or a single value (pass an empty max vector)  -> fixed value

BField = 3.8
from CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi import siStripLorentzAngleFakeESSource
# TIB min and max
siStripLorentzAngleFakeESSource.TIB_EstimatedValuesMin = cms.vdouble(0.0/BField, 0.0/BField, 0.0/BField, 0.0/BField)
siStripLorentzAngleFakeESSource.TIB_EstimatedValuesMax = cms.vdouble(0.10/BField, 0.10/BField, 0.10/BField, 0.10/BField)
# TIB errors
siStripLorentzAngleFakeESSource.TIB_PerCent_Errs       = cms.vdouble(0.,    0.,    0.,    0.)
# TOB min and max
siStripLorentzAngleFakeESSource.TOB_EstimatedValuesMin = cms.vdouble(0.0/BField, 0.0/BField, 0.0/BField, 0.0/BField, 0.0/BField, 0.0/BField)
siStripLorentzAngleFakeESSource.TOB_EstimatedValuesMax = cms.vdouble(0.12/BField, 0.12/BField, 0.12/BField, 0.12/BField, 0.12/BField, 0.12/BField)
# TOB errors
siStripLorentzAngleFakeESSource.TOB_PerCent_Errs       = cms.vdouble(0.,    0.,    0.,    0.,    0.,    0.)

process.p1 = cms.Path(process.siStripLorentzAngleDummyDBWriter)

