import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

#process.PhotonConversionMVAComputerSave = cms.ESSource("PhotonConversionMVAComputerSave",
process.PhotonConversionMVAComputerFileSource = cms.ESSource("PhotonConversionMVAComputerFileSource",
        label = cms.string('test.mva'),
        )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
        DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0) ),
        BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
        timetype = cms.untracked.string('runnumber'),
        connect = cms.string('sqlite_file:localconditions.db'),
        toPut = cms.VPSet(cms.PSet(
            record = cms.string('PhotonConversionMVAComputerRcd'),
            tag = cms.string('some_pooldb_tag')
            ))
        )

process.PhotonConversionMVAComputerSave = cms.EDAnalyzer("PhotonConversionMVAComputerSave",
        toPut = cms.vstring(),
        toCopy = cms.vstring("label")
        )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.outpath = cms.EndPath(process.PhotonConversionMVAComputerSave)
