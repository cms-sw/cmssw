import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:test.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelQualityRcd'),
        tag = cms.string('SiPixelBadModule_test')
    ))
)

process.prod = cms.EDFilter("SiPixelBadModuleByHandBuilder",
    BadModuleList = cms.untracked.VPSet(cms.PSet(
        errortype = cms.string('whole'),
        detid = cms.uint32(302197784)
    ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(302195232)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(302123296)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(302127136)
        ), 
        cms.PSet(
            errortype = cms.string('tbmA'),
            detid = cms.uint32(302125076)
        ), 
        cms.PSet(
            errortype = cms.string('tbmB'),
            detid = cms.uint32(302126364)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(302188552)
        ), 
        cms.PSet(
            errortype = cms.string('tbmA'),
            detid = cms.uint32(302121992)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(302126596)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344074500)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344074504)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344074508)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344074512)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344074756)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344074760)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344074764)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344075524)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344075528)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344075532)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344075536)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344075780)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344075784)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344075788)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344076548)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344076552)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344076556)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344076560)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344076804)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344076808)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344076812)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344005128)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344020236)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344020240)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344020488)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344020492)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344019212)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344019216)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344019464)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344019468)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344018188)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344018192)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344018440)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344018444)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344014340)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344014344)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344014348)
        )),
    Record = cms.string('SiPixelQualityRcd'),
    SinceAppendMode = cms.bool(True),
    IOVMode = cms.string('Run'),
    printDebug = cms.untracked.bool(True),
    doStoreOnDB = cms.bool(True)
)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
#process.ep = cms.EndPath(process.print)


