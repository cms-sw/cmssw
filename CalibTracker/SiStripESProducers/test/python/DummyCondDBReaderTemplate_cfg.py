# Template used by the UploadAll.py scripts to generate the cfgs for the DummyPrinter
# The only templated value is TAGNAME (e.g TAGNAME -> SiStripApvGain).
# To generate manually a cfg from this file please do for example:
# cat DummyCondDBReaderTemplate_cfg.py | sed -e "s@TAGNAME@SiStripApvGain@" | sed -e "s@RCDNAME@SiStripApvGainRcd@" > DummyCondDBReader_SiStripApvGain_cfg.py
import FWCore.ParameterSet.Config as cms

process = cms.Process("Reader")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('reader'),
    files = cms.untracked.PSet(
        TAGNAMEReader_Ideal = cms.untracked.PSet(

        )
    ),
    threshold = cms.untracked.string('INFO')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RCDNAME'),
        tag = cms.string('TAGNAME_Ideal_31X')
    ))
)

process.reader = cms.EDAnalyzer("TAGNAMEDummyPrinter")


process.p1 = cms.Path(process.reader)


