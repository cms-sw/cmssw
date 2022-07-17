import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripBadStripFromASCIIFile=dict()  
process.MessageLogger.SiStripBadStrip=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("DEBUG"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripBadStripFromASCIIFile = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiStripBadStrip = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:SiStripConstructionDefectsDBFile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStrip'),
        tag = cms.string('SiStripBadChannel_v1')
    ))
)

from CondTools.SiStrip.siStripBadStripFromASCIIFile_cfi import siStripBadStripFromASCIIFile
process.prod = siStripBadStripFromASCIIFile.clone(Record = cms.string('SiStripBadStrip'),
                                                  printDebug = cms.bool(False),
                                                  IOVMode = cms.string('Run'),
                                                  SinceAppendMode = cms.bool(True),
                                                  doStoreOnDB = cms.bool(True),
                                                  file = cms.FileInPath('CondTools/SiStrip/data/DefectsFromConstructionDB.dat'))

process.p = cms.Path(process.prod)


