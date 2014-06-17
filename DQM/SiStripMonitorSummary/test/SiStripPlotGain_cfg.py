import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")
process.MessageLogger = cms.Service("MessageLogger",
                                    out = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('out')
                                    )

process.source = cms.Source("EmptyIOVSource",
                            firstValue = cms.uint64(109574),
                            lastValue = cms.uint64(109574),
                            timetype = cms.string('runnumber'),
                            interval = cms.uint64(1)
                            )

# the DB Geometry is NOT used because in this cfg only one tag is taken from the DB and no GT is used. To be fixed if this is a problem
process.load('Configuration.Geometry.GeometryExtended_cff')
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.poolDBESSource = cms.ESSource(
                                      "PoolDBESSource",
                                      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                      DBParameters = cms.PSet(messageLevel = cms.untracked.int32(2),
                                                              authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
                                                              ),
                                      timetype = cms.untracked.string('runnumber'),
                                      connect = cms.string('frontier://FrontierProd/CMS_COND_31X_STRIP'),
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                                                 tag = cms.string('SiStripApvGain_GR09_31X_v1_hlt')
                                                                 )
                                                        )
                                      )


process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
process.analysis = cms.EDAnalyzer("SiStripPlotGain")


process.p = cms.Path(process.analysis)

