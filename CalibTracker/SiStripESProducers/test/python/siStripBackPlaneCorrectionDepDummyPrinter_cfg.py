# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("Reader")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('BackPlaneCorrectionReaderSummary'),
    files = cms.untracked.PSet(
        BackPlaneCorrectionReader = cms.untracked.PSet(

        )
    ),
    threshold = cms.untracked.string('DEBUG')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(200000)
)

#DBESSource
process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toGet = cms.VPSet(
       cms.PSet(
           record = cms.string('SiStripBackPlaneCorrectionRcd'),
           tag = cms.string('SiStripBackPlaneCorrection_deco_31X'),
           label = cms.untracked.string('deconvolution')
       ),
       cms.PSet(
           record = cms.string('SiStripBackPlaneCorrectionRcd'),
           tag = cms.string('SiStripBackPlaneCorrection_peak_31X'),
           label = cms.untracked.string('peak')
       ),
    )
)

#Latency producer
process.load("CalibTracker.SiStripESProducers.fake.SiStripLatencyFakeESSource_cfi")
from CalibTracker.SiStripESProducers.fake.SiStripLatencyFakeESSource_cfi import siStripLatencyFakeESSource
#siStripLatencyFakeESSource.latency = 255
#siStripLatencyFakeESSource.mode = 0
siStripLatencyFakeESSource.latency = 143
siStripLatencyFakeESSource.mode = 47
# siStripLatencyFakeESSource.latency = 146
# siStripLatencyFakeESSource.mode = 37

#Dependent ESSource
process.siStripBackPlaneCorrectionDepESProducer = cms.ESProducer("SiStripBackPlaneCorrectionDepESProducer",
     LatencyRecord =   cms.PSet(
            record = cms.string('SiStripLatencyRcd'),
            label = cms.untracked.string('')
            ),
        BackPlaneCorrectionPeakMode = cms.PSet(
            record = cms.string('SiStripBackPlaneCorrectionRcd'),
            label = cms.untracked.string('peak')
            ),
        BackPlaneCorrectionDeconvMode = cms.PSet(
            record = cms.string('SiStripBackPlaneCorrectionRcd'),
            label = cms.untracked.string('deconvolution')
            )
)

process.reader = cms.EDAnalyzer("SiStripBackPlaneCorrectionDepDummyPrinter")
                              
process.p1 = cms.Path(process.reader)


