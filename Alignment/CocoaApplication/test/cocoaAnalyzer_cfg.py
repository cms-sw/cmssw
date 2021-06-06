import FWCore.ParameterSet.Config as cms

process = cms.Process("TestCocoa")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('cocoa')
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG')
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# A data source must always be defined. We don't need it, so here's a dummy one.
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)


# dd4hep-based geometry
process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Alignment/CocoaApplication/test/cmsCocoaTable2DWithMirror.xml'),
                                            appendToDataLabel = cms.string('')
                                            )
                                            
process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                 appendToDataLabel = cms.string('')
                                                )

                                                                  
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:OpticalAlignments.db'


# Read DB: this is used to correct internal geometry with geo from DB.
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('OpticalAlignmentsRcd'),
        tag = cms.string("OpticalAlignmentsRcdInput")
    )),
)


# Write COCOA output to DB ('Output tag')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
    	cms.PSet(
        	record = cms.string('OpticalAlignmentsRcd'),
        	tag = cms.string('OpticalAlignmentsRcdOutput')
    	), 
    	cms.PSet(
        	record = cms.string('DTAlignmentRcd'),
        	tag = cms.string('DTAlignmentRcdOutput')
    	),
    	cms.PSet(
        	record = cms.string('DTAlignmentErrorExtendedRcd'),
        	tag = cms.string('DTAlignmentErrorExtendedRcdOutput')
    	),
    	cms.PSet(
        	record = cms.string('CSCAlignmentRcd'),
        	tag = cms.string('CSCAlignmentRcdOutput')
    	),
    	cms.PSet(
        	record = cms.string('CSCAlignmentErrorExtendedRcd'),
        	tag = cms.string('CSCAlignmentErrorExtendedRcdOutput')
    	),
    )
)


# Run COCOA                                                
process.cocoa = cms.EDAnalyzer('CocoaAnalyzer',
				maxEvents = cms.int32(1),
				cocoaDaqRootFile = cms.string("cocoaDaqTest.root")
                              )

process.p = cms.Path(process.cocoa)

