
import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")
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
                            firstValue = cms.uint64(108597),
                            lastValue = cms.uint64(108597),
                            timetype = cms.string('runnumber'),
                            interval = cms.uint64(1)
                            )

# the DB Geometry is NOT used because in this cfg only one tag is taken from the DB and no GT is used. To be fixed if this is a problem
process.load('Configuration.Geometry.GeometryExtended_cff')
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.trackerGeometry.applyAlignment = False

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.poolDBESSource = cms.ESSource(
                                      "PoolDBESSource",
                                      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                      DBParameters = cms.PSet(messageLevel = cms.untracked.int32(2),
                                                              authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
                                                              ),
                                      timetype = cms.untracked.string('runnumber'),
                                      connect = cms.string('frontier://FrontierProd/CMS_COND_31X_STRIP'),
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('SiStripNoisesRcd'),
                                                                 tag = cms.string('SiStripNoise_GR09_31X_v1_hlt')
                                                                 ),
                                                        cms.PSet(record = cms.string('SiStripBadChannelRcd'),
                                                                 tag = cms.string('SiStripBadComponents_OfflineAnalysis_HotStrips_GR09_31X_v1_offline')
                                                                 )
                                                        )
                                      )
    

process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
                                                  ReduceGranularity = cms.bool(False),
                                                  PrintDebugOutput = cms.bool(False),
                                                  UseEmptyRunInfo = cms.bool(False),
                                                  ListOfRecordToMerge = cms.VPSet(cms.PSet(record = cms.string('SiStripBadChannelRcd'),
                                                                                           tag = cms.string('')
                                                                                           )
                                                                                  )
                                                  )


from CalibTracker.SiStripQuality.siStripQualityStatistics_cfi import siStripQualityStatistics
process.stat = siStripQualityStatistics.clone(
        #TkMapFileName = cms.untracked.string('TkMaps/TkMapBadComponents_full.png'),
        )
process.analysis = cms.EDAnalyzer("SiStripCorrelateBadStripAndNoise")


process.p = cms.Path(process.stat+process.analysis)

