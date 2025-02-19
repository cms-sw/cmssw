import FWCore.ParameterSet.Config as cms

process = cms.Process("HitEff")

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(newrun),
    lastValue = cms.uint64(newrun),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

process.SiStripHitEff = cms.EDFilter("SiStripHitEffFromCalibTree",
    CalibTreeFilename = cms.string('rfio:newfilelocation'),
    Threshold         = cms.double(0.1),
    nModsMin          = cms.int32(25),
    doSummary         = cms.int32(0),
    ResXSig           = cms.untracked.double(5),
    SinceAppendMode   = cms.bool(True),
    IOVMode           = cms.string('Run'),
    Record            = cms.string('SiStripBadStrip'),
    doStoreOnDB       = cms.bool(True)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStrip'),
        tag = cms.string('SiStripHitEffBadModules')
    ))
)

process.TFileService = cms.Service("TFileService",
        fileName = cms.string('SiStripHitEffHistos_runnewrun.root')  
)

process.allPath = cms.Path(process.SiStripHitEff)

