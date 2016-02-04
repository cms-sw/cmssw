import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")
process.MessageLogger = cms.Service("MessageLogger",
                                    out = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('out')
                                    )

process.source = cms.Source("EmptyIOVSource",
                            firstValue = cms.uint64(insertFirstRun),
                            lastValue = cms.uint64(insertSecondRun),
                            timetype = cms.string('runnumber'),
                            interval = cms.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.poolDBESSource = cms.ESSource(
                                      "PoolDBESSource",
                                      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                      DBParameters = cms.PSet(messageLevel = cms.untracked.int32(2),
                                                              authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
                                                              ),
                                      timetype = cms.untracked.string('runnumber'),
                                      connect = cms.string('frontier://insertFrontier/insertAccount'),
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('SiStripNoisesRcd'),
                                                                 tag = cms.string('insertNoiseTag')
                                                                 ),
                                                        cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                                                 tag = cms.string('insertGainTag')
                                                                 )
                                                        )
                                      )


process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
process.analysis = cms.EDAnalyzer("SiStripCorrelateNoise")


process.p = cms.Path(process.analysis)

