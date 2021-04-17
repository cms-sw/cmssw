# The following comments couldn't be translated into the new config version:

# upload to database 

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("DeDxMipReader")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring(''),
    fedcablingReader = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    files = cms.untracked.PSet(
        SiStripDeDx3DReader = cms.untracked.PSet(

        )
    )
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
    connect = cms.string('sqlite_file:dummy2.db'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripDeDxProton_3D_Rcd'),
        tag = cms.string('SiStripDeDxProton_3D_30X')
    ))
)

process.test = cms.EDAnalyzer("SiStripDeDx3DReader")


process.p1 = cms.Path(process.test)


