import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
#process.load("Configuration.StandardSequences.GeometryIdeal_cff")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptyIOVSource",
                                lastValue = cms.uint64(1),
                                timetype = cms.string('runnumber'),
                                firstValue = cms.uint64(1),
                                interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),

                                          timetype = cms.untracked.string('runnumber'),
                                          connect = cms.string('sqlite_file:test.db'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('DTRecoGeometryRcd'),tag = cms.string('XMLFILE_TEST_01'))
                                                            )
                                          )

process.dtload = cms.EDFilter("DTRecoIdealDBLoader")

process.p1 = cms.Path(process.dtload)
