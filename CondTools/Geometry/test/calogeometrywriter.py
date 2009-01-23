import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloGeometryWriter")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load('Configuration/StandardSequences/GeometryIdeal_cff')

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.CaloGeometryWriter = cms.EDAnalyzer("PCaloGeometryBuilder")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                          timetype = cms.untracked.string('runnumber'),
                                          connect = cms.string('sqlite_file:myfile.db'),
                                          toPut = cms.VPSet(
    cms.PSet(record = cms.string('PEcalBarrelRcd'),   tag = cms.string('TEST02')),
    cms.PSet(record = cms.string('PEcalEndcapRcd'),   tag = cms.string('TEST03')),
    cms.PSet(record = cms.string('PEcalPreshowerRcd'),tag = cms.string('TEST04')),
    cms.PSet(record = cms.string('PHcalRcd'),         tag = cms.string('TEST05')) )
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.CaloGeometryWriter)

