# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("Builder")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring(''),
    LorentzAngleBuilder = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('LorentzAngleBuilder.log')
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
    connect = cms.string('sqlite_file:dbfile.db'),
    timetype = cms.untracked.string('runnumber'),    
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripLorentzAngleRcd'),
        tag = cms.string('SiStripLorentzAngle_Ideal_31X')
    ))
)

process.SiStripLorentzAngleGenerator.TIB_EstimatedValue = cms.double(0.01784)
process.SiStripLorentzAngleGenerator.TOB_EstimatedValue = cms.double(0.02315)
process.SiStripLorentzAngleGenerator.TIB_PerCent_Err=cms.double(20.)
process.SiStripLorentzAngleGenerator.TOB_PerCent_Err=cms.double(10.)

process.p1 = cms.Path(process.siStripLorentzAngleDummyDBWriter)


