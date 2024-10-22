import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")

process.load("Configuration.Geometry.GeometryExtended2017_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
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
    connect = cms.string('sqlite_file:prova.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelQualityFromDbRcd'),
        tag = cms.string('SiPixelQuality_v07_mc')
    ))
)

process.prod = cms.EDAnalyzer("SiPixelBadModuleByHandBuilder",
    ROCListFile = cms.untracked.string(""), 
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
            errortype = cms.string('whole'),
            detid = cms.uint32(302121992)
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(302126596)
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
        ), 
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344019460)
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
            detid = cms.uint32(344078852)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344078856)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344078860)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344078596)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344078600)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344078604)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344078608)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344077572)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344077576)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344077580)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344077584)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344079620)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344079624)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344079628)
        ),
        cms.PSet(
            errortype = cms.string('whole'),
            detid = cms.uint32(344079632)
        ),
	cms.PSet(
	    errortype = cms.string('whole'),
	    detid = cms.uint32(302059800)
	),
         cms.PSet(
            errortype = cms.string('tbmA'),
            detid = cms.uint32(302125060)
        )),
 
    Record = cms.string('SiPixelQualityFromDbRcd'),
    SinceAppendMode = cms.bool(True),
    IOVMode = cms.string('Run'),
    printDebug = cms.untracked.bool(True),
    doStoreOnDB = cms.bool(True)
)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
#process.ep = cms.EndPath(process.print)


