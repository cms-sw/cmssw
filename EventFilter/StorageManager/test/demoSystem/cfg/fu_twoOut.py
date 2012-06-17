import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("FU")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True),
                                     makeTriggerResults = cms.untracked.bool(True),
                                     Rethrow = cms.untracked.vstring('ProductNotFound',
                                                                     'TooManyProducts',
                                                                     'TooFewProducts')
                                     )

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout','log4cplus'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
                                    log4cplus = cms.untracked.PSet(INFO = cms.untracked.PSet(reportEvery = cms.untracked.int32(250)),
                                                                   threshold = cms.untracked.string('INFO')
                                                                   )
                                    )

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.SiteLocalConfigService = cms.Service("SiteLocalConfigService")

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.DQMStore = cms.Service("DQMStore")

process.source = cms.Source("DaqSource",
                            readerPluginName = cms.untracked.string('FUShmReader'),
                            evtsPerLS = cms.untracked.uint32(10000)
                            )

process.errorInjector = cms.EDAnalyzer("ExceptionGenerator")
process.p0 = cms.Path(process.errorInjector)

process.m2b20 = cms.EDProducer("StreamThingProducer",
                               array_size = cms.int32(20),
                               instance_count = cms.int32(2)
                               )

process.m3b10 = cms.EDProducer("StreamThingProducer",
                               array_size = cms.int32(10),
                               instance_count = cms.int32(3)
                               )

process.m4b15000 = cms.EDProducer("StreamThingProducer",
                               array_size = cms.int32(15000),
                               instance_count = cms.int32(4)
                               )

process.m5b20000 = cms.EDProducer("StreamThingProducer",
                               array_size = cms.int32(20000),
                               instance_count = cms.int32(5)
                               )

process.m6b25 = cms.EDProducer("StreamThingProducer",
                               array_size = cms.int32(25),
                               instance_count = cms.int32(6)
                               )

process.prescaleBy3 = cms.EDFilter("Prescaler",
                                   prescaleFactor = cms.int32(3),
                                   prescaleOffset = cms.int32(0)
                                   )

process.prescaleBy5 = cms.EDFilter("Prescaler",
                                   prescaleFactor = cms.int32(5),
                                   prescaleOffset = cms.int32(0)
                                   )

process.prescaleBy7 = cms.EDFilter("Prescaler",
                                   prescaleFactor = cms.int32(7),
                                   prescaleOffset = cms.int32(0)
                                   )

process.prescaleBy11 = cms.EDFilter("Prescaler",
                                    prescaleFactor = cms.int32(11),
                                    prescaleOffset = cms.int32(0)
                                    )

#process.prescaleBy1000 = cms.EDFilter("Prescaler",
#                                      prescaleFactor = cms.int32(1000),
#                                      prescaleOffset = cms.int32(0)
#                                      )


process.DiMuon = cms.Path(process.prescaleBy5 * process.m2b20 * process.m5b20000)

process.DiElectron = cms.Path(process.prescaleBy7 * process.m6b25)

process.CalibPath = cms.Path(process.prescaleBy11 * process.m4b15000)

process.HighPT = cms.Path(process.prescaleBy3 + process.m3b10)

process.playbackPath4DQM = cms.Path(process.prescaleBy3 + process.m3b10)


process.PhysicsOModule = cms.OutputModule("ShmStreamConsumer",
                                          use_compression = cms.untracked.bool(True),
                                          SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('DiMuon','CalibPath','DiElectron','HighPT') ),
                                          outputCommands = cms.untracked.vstring('keep *','drop *_m4b15000_*_*')
                                          )
process.end1 = cms.EndPath(process.PhysicsOModule)

process.hltOutputDQM = cms.OutputModule("ShmStreamConsumer",
                                        use_compression = cms.untracked.bool(True),
                                        SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('playbackPath4DQM') ),
                                        outputCommands = cms.untracked.vstring('keep *','drop *_m5b20000_*_*')
                                        )
process.end2 = cms.EndPath(process.hltOutputDQM)


process.DQMTester = cms.EDAnalyzer("SMDQMSourceExample")
process.DQMpath = cms.Path(process.DQMTester)

process.FUShmDQMOutputService = cms.Service("FUShmDQMOutputService",
                                            initialMessageBufferSize = cms.untracked.int32(1000000),
                                            lumiSectionsPerUpdate = cms.double(1.0),
                                            useCompression = cms.bool(True),
                                            compressionLevel = cms.int32(1)
                                            )


if 'SMDEV_BIG_HLT_CONFIG' in os.environ.keys() and \
    os.environ['SMDEV_BIG_HLT_CONFIG'] == "1":

    process.bigCfgModule000 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath000 = cms.Path(process.bigCfgModule000)

    process.bigCfgModule001 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath001 = cms.Path(process.bigCfgModule001)

    process.bigCfgModule002 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath002 = cms.Path(process.bigCfgModule002)

    process.bigCfgModule003 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath003 = cms.Path(process.bigCfgModule003)

    process.bigCfgModule004 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath004 = cms.Path(process.bigCfgModule004)

    process.bigCfgModule005 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath005 = cms.Path(process.bigCfgModule005)

    process.bigCfgModule006 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath006 = cms.Path(process.bigCfgModule006)

    process.bigCfgModule007 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath007 = cms.Path(process.bigCfgModule007)

    process.bigCfgModule008 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath008 = cms.Path(process.bigCfgModule008)

    process.bigCfgModule009 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath009 = cms.Path(process.bigCfgModule009)


    process.bigCfgModule010 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath010 = cms.Path(process.bigCfgModule010)

    process.bigCfgModule011 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath011 = cms.Path(process.bigCfgModule011)

    process.bigCfgModule012 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath012 = cms.Path(process.bigCfgModule012)

    process.bigCfgModule013 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath013 = cms.Path(process.bigCfgModule013)

    process.bigCfgModule014 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath014 = cms.Path(process.bigCfgModule014)

    process.bigCfgModule015 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath015 = cms.Path(process.bigCfgModule015)

    process.bigCfgModule016 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath016 = cms.Path(process.bigCfgModule016)

    process.bigCfgModule017 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath017 = cms.Path(process.bigCfgModule017)

    process.bigCfgModule018 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath018 = cms.Path(process.bigCfgModule018)

    process.bigCfgModule019 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath019 = cms.Path(process.bigCfgModule019)


    process.bigCfgModule020 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath020 = cms.Path(process.bigCfgModule020)

    process.bigCfgModule021 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath021 = cms.Path(process.bigCfgModule021)

    process.bigCfgModule022 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath022 = cms.Path(process.bigCfgModule022)

    process.bigCfgModule023 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath023 = cms.Path(process.bigCfgModule023)

    process.bigCfgModule024 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath024 = cms.Path(process.bigCfgModule024)

    process.bigCfgModule025 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath025 = cms.Path(process.bigCfgModule025)

    process.bigCfgModule026 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath026 = cms.Path(process.bigCfgModule026)

    process.bigCfgModule027 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath027 = cms.Path(process.bigCfgModule027)

    process.bigCfgModule028 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath028 = cms.Path(process.bigCfgModule028)

    process.bigCfgModule029 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath029 = cms.Path(process.bigCfgModule029)


    process.bigCfgModule030 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath030 = cms.Path(process.bigCfgModule030)

    process.bigCfgModule031 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath031 = cms.Path(process.bigCfgModule031)

    process.bigCfgModule032 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath032 = cms.Path(process.bigCfgModule032)

    process.bigCfgModule033 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath033 = cms.Path(process.bigCfgModule033)

    process.bigCfgModule034 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath034 = cms.Path(process.bigCfgModule034)

    process.bigCfgModule035 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath035 = cms.Path(process.bigCfgModule035)

    process.bigCfgModule036 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath036 = cms.Path(process.bigCfgModule036)

    process.bigCfgModule037 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath037 = cms.Path(process.bigCfgModule037)

    process.bigCfgModule038 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath038 = cms.Path(process.bigCfgModule038)

    process.bigCfgModule039 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath039 = cms.Path(process.bigCfgModule039)


    process.bigCfgModule040 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath040 = cms.Path(process.bigCfgModule040)

    process.bigCfgModule041 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath041 = cms.Path(process.bigCfgModule041)

    process.bigCfgModule042 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath042 = cms.Path(process.bigCfgModule042)

    process.bigCfgModule043 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath043 = cms.Path(process.bigCfgModule043)

    process.bigCfgModule044 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath044 = cms.Path(process.bigCfgModule044)

    process.bigCfgModule045 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath045 = cms.Path(process.bigCfgModule045)

    process.bigCfgModule046 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath046 = cms.Path(process.bigCfgModule046)

    process.bigCfgModule047 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath047 = cms.Path(process.bigCfgModule047)

    process.bigCfgModule048 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath048 = cms.Path(process.bigCfgModule048)

    process.bigCfgModule049 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath049 = cms.Path(process.bigCfgModule049)


    process.bigCfgModule050 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath050 = cms.Path(process.bigCfgModule050)

    process.bigCfgModule051 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath051 = cms.Path(process.bigCfgModule051)

    process.bigCfgModule052 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath052 = cms.Path(process.bigCfgModule052)

    process.bigCfgModule053 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath053 = cms.Path(process.bigCfgModule053)

    process.bigCfgModule054 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath054 = cms.Path(process.bigCfgModule054)

    process.bigCfgModule055 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath055 = cms.Path(process.bigCfgModule055)

    process.bigCfgModule056 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath056 = cms.Path(process.bigCfgModule056)

    process.bigCfgModule057 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath057 = cms.Path(process.bigCfgModule057)

    process.bigCfgModule058 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath058 = cms.Path(process.bigCfgModule058)

    process.bigCfgModule059 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath059 = cms.Path(process.bigCfgModule059)


    process.bigCfgModule060 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath060 = cms.Path(process.bigCfgModule060)

    process.bigCfgModule061 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath061 = cms.Path(process.bigCfgModule061)

    process.bigCfgModule062 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath062 = cms.Path(process.bigCfgModule062)

    process.bigCfgModule063 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath063 = cms.Path(process.bigCfgModule063)

    process.bigCfgModule064 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath064 = cms.Path(process.bigCfgModule064)

    process.bigCfgModule065 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath065 = cms.Path(process.bigCfgModule065)

    process.bigCfgModule066 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath066 = cms.Path(process.bigCfgModule066)

    process.bigCfgModule067 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath067 = cms.Path(process.bigCfgModule067)

    process.bigCfgModule068 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath068 = cms.Path(process.bigCfgModule068)

    process.bigCfgModule069 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath069 = cms.Path(process.bigCfgModule069)


    process.bigCfgModule070 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath070 = cms.Path(process.bigCfgModule070)

    process.bigCfgModule071 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath071 = cms.Path(process.bigCfgModule071)

    process.bigCfgModule072 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath072 = cms.Path(process.bigCfgModule072)

    process.bigCfgModule073 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath073 = cms.Path(process.bigCfgModule073)

    process.bigCfgModule074 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath074 = cms.Path(process.bigCfgModule074)

    process.bigCfgModule075 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath075 = cms.Path(process.bigCfgModule075)

    process.bigCfgModule076 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath076 = cms.Path(process.bigCfgModule076)

    process.bigCfgModule077 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath077 = cms.Path(process.bigCfgModule077)

    process.bigCfgModule078 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath078 = cms.Path(process.bigCfgModule078)

    process.bigCfgModule079 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath079 = cms.Path(process.bigCfgModule079)


    process.bigCfgModule080 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath080 = cms.Path(process.bigCfgModule080)

    process.bigCfgModule081 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath081 = cms.Path(process.bigCfgModule081)

    process.bigCfgModule082 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath082 = cms.Path(process.bigCfgModule082)

    process.bigCfgModule083 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath083 = cms.Path(process.bigCfgModule083)

    process.bigCfgModule084 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath084 = cms.Path(process.bigCfgModule084)

    process.bigCfgModule085 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath085 = cms.Path(process.bigCfgModule085)

    process.bigCfgModule086 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath086 = cms.Path(process.bigCfgModule086)

    process.bigCfgModule087 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath087 = cms.Path(process.bigCfgModule087)

    process.bigCfgModule088 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath088 = cms.Path(process.bigCfgModule088)

    process.bigCfgModule089 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath089 = cms.Path(process.bigCfgModule089)


    process.bigCfgModule090 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath090 = cms.Path(process.bigCfgModule090)

    process.bigCfgModule091 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath091 = cms.Path(process.bigCfgModule091)

    process.bigCfgModule092 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath092 = cms.Path(process.bigCfgModule092)

    process.bigCfgModule093 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath093 = cms.Path(process.bigCfgModule093)

    process.bigCfgModule094 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath094 = cms.Path(process.bigCfgModule094)

    process.bigCfgModule095 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath095 = cms.Path(process.bigCfgModule095)

    process.bigCfgModule096 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath096 = cms.Path(process.bigCfgModule096)

    process.bigCfgModule097 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath097 = cms.Path(process.bigCfgModule097)

    process.bigCfgModule098 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath098 = cms.Path(process.bigCfgModule098)

    process.bigCfgModule099 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath099 = cms.Path(process.bigCfgModule099)


    process.bigCfgModule100 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath100 = cms.Path(process.bigCfgModule100)

    process.bigCfgModule101 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath101 = cms.Path(process.bigCfgModule101)

    process.bigCfgModule102 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath102 = cms.Path(process.bigCfgModule102)

    process.bigCfgModule103 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath103 = cms.Path(process.bigCfgModule103)

    process.bigCfgModule104 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath104 = cms.Path(process.bigCfgModule104)

    process.bigCfgModule105 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath105 = cms.Path(process.bigCfgModule105)

    process.bigCfgModule106 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath106 = cms.Path(process.bigCfgModule106)

    process.bigCfgModule107 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath107 = cms.Path(process.bigCfgModule107)

    process.bigCfgModule108 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath108 = cms.Path(process.bigCfgModule108)

    process.bigCfgModule109 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath109 = cms.Path(process.bigCfgModule109)


    process.bigCfgModule110 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath110 = cms.Path(process.bigCfgModule110)

    process.bigCfgModule111 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath111 = cms.Path(process.bigCfgModule111)

    process.bigCfgModule112 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath112 = cms.Path(process.bigCfgModule112)

    process.bigCfgModule113 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath113 = cms.Path(process.bigCfgModule113)

    process.bigCfgModule114 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath114 = cms.Path(process.bigCfgModule114)

    process.bigCfgModule115 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath115 = cms.Path(process.bigCfgModule115)

    process.bigCfgModule116 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath116 = cms.Path(process.bigCfgModule116)

    process.bigCfgModule117 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath117 = cms.Path(process.bigCfgModule117)

    process.bigCfgModule118 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath118 = cms.Path(process.bigCfgModule118)

    process.bigCfgModule119 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath119 = cms.Path(process.bigCfgModule119)


    process.bigCfgModule120 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath120 = cms.Path(process.bigCfgModule120)

    process.bigCfgModule121 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath121 = cms.Path(process.bigCfgModule121)

    process.bigCfgModule122 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath122 = cms.Path(process.bigCfgModule122)

    process.bigCfgModule123 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath123 = cms.Path(process.bigCfgModule123)

    process.bigCfgModule124 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath124 = cms.Path(process.bigCfgModule124)

    process.bigCfgModule125 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath125 = cms.Path(process.bigCfgModule125)

    process.bigCfgModule126 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath126 = cms.Path(process.bigCfgModule126)

    process.bigCfgModule127 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath127 = cms.Path(process.bigCfgModule127)

    process.bigCfgModule128 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath128 = cms.Path(process.bigCfgModule128)

    process.bigCfgModule129 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath129 = cms.Path(process.bigCfgModule129)


    process.bigCfgModule130 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath130 = cms.Path(process.bigCfgModule130)

    process.bigCfgModule131 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath131 = cms.Path(process.bigCfgModule131)

    process.bigCfgModule132 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath132 = cms.Path(process.bigCfgModule132)

    process.bigCfgModule133 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath133 = cms.Path(process.bigCfgModule133)

    process.bigCfgModule134 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath134 = cms.Path(process.bigCfgModule134)

    process.bigCfgModule135 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath135 = cms.Path(process.bigCfgModule135)

    process.bigCfgModule136 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath136 = cms.Path(process.bigCfgModule136)

    process.bigCfgModule137 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath137 = cms.Path(process.bigCfgModule137)

    process.bigCfgModule138 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath138 = cms.Path(process.bigCfgModule138)

    process.bigCfgModule139 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath139 = cms.Path(process.bigCfgModule139)


    process.bigCfgModule140 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath140 = cms.Path(process.bigCfgModule140)

    process.bigCfgModule141 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath141 = cms.Path(process.bigCfgModule141)

    process.bigCfgModule142 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath142 = cms.Path(process.bigCfgModule142)

    process.bigCfgModule143 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath143 = cms.Path(process.bigCfgModule143)

    process.bigCfgModule144 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath144 = cms.Path(process.bigCfgModule144)

    process.bigCfgModule145 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath145 = cms.Path(process.bigCfgModule145)

    process.bigCfgModule146 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath146 = cms.Path(process.bigCfgModule146)

    process.bigCfgModule147 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath147 = cms.Path(process.bigCfgModule147)

    process.bigCfgModule148 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath148 = cms.Path(process.bigCfgModule148)

    process.bigCfgModule149 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath149 = cms.Path(process.bigCfgModule149)


    process.bigCfgModule150 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath150 = cms.Path(process.bigCfgModule150)

    process.bigCfgModule151 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath151 = cms.Path(process.bigCfgModule151)

    process.bigCfgModule152 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath152 = cms.Path(process.bigCfgModule152)

    process.bigCfgModule153 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath153 = cms.Path(process.bigCfgModule153)

    process.bigCfgModule154 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath154 = cms.Path(process.bigCfgModule154)

    process.bigCfgModule155 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath155 = cms.Path(process.bigCfgModule155)

    process.bigCfgModule156 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath156 = cms.Path(process.bigCfgModule156)

    process.bigCfgModule157 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath157 = cms.Path(process.bigCfgModule157)

    process.bigCfgModule158 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath158 = cms.Path(process.bigCfgModule158)

    process.bigCfgModule159 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath159 = cms.Path(process.bigCfgModule159)


    process.bigCfgModule160 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath160 = cms.Path(process.bigCfgModule160)

    process.bigCfgModule161 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath161 = cms.Path(process.bigCfgModule161)

    process.bigCfgModule162 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath162 = cms.Path(process.bigCfgModule162)

    process.bigCfgModule163 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath163 = cms.Path(process.bigCfgModule163)

    process.bigCfgModule164 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath164 = cms.Path(process.bigCfgModule164)

    process.bigCfgModule165 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath165 = cms.Path(process.bigCfgModule165)

    process.bigCfgModule166 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath166 = cms.Path(process.bigCfgModule166)

    process.bigCfgModule167 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath167 = cms.Path(process.bigCfgModule167)

    process.bigCfgModule168 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath168 = cms.Path(process.bigCfgModule168)

    process.bigCfgModule169 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath169 = cms.Path(process.bigCfgModule169)


    process.bigCfgModule170 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath170 = cms.Path(process.bigCfgModule170)

    process.bigCfgModule171 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath171 = cms.Path(process.bigCfgModule171)

    process.bigCfgModule172 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath172 = cms.Path(process.bigCfgModule172)

    process.bigCfgModule173 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath173 = cms.Path(process.bigCfgModule173)

    process.bigCfgModule174 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath174 = cms.Path(process.bigCfgModule174)

    process.bigCfgModule175 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath175 = cms.Path(process.bigCfgModule175)

    process.bigCfgModule176 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath176 = cms.Path(process.bigCfgModule176)

    process.bigCfgModule177 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath177 = cms.Path(process.bigCfgModule177)

    process.bigCfgModule178 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath178 = cms.Path(process.bigCfgModule178)

    process.bigCfgModule179 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath179 = cms.Path(process.bigCfgModule179)


    process.bigCfgModule180 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath180 = cms.Path(process.bigCfgModule180)

    process.bigCfgModule181 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath181 = cms.Path(process.bigCfgModule181)

    process.bigCfgModule182 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath182 = cms.Path(process.bigCfgModule182)

    process.bigCfgModule183 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath183 = cms.Path(process.bigCfgModule183)

    process.bigCfgModule184 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath184 = cms.Path(process.bigCfgModule184)

    process.bigCfgModule185 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath185 = cms.Path(process.bigCfgModule185)

    process.bigCfgModule186 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath186 = cms.Path(process.bigCfgModule186)

    process.bigCfgModule187 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath187 = cms.Path(process.bigCfgModule187)

    process.bigCfgModule188 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath188 = cms.Path(process.bigCfgModule188)

    process.bigCfgModule189 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath189 = cms.Path(process.bigCfgModule189)


    process.bigCfgModule190 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath190 = cms.Path(process.bigCfgModule190)

    process.bigCfgModule191 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath191 = cms.Path(process.bigCfgModule191)

    process.bigCfgModule192 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath192 = cms.Path(process.bigCfgModule192)

    process.bigCfgModule193 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath193 = cms.Path(process.bigCfgModule193)

    process.bigCfgModule194 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath194 = cms.Path(process.bigCfgModule194)

    process.bigCfgModule195 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath195 = cms.Path(process.bigCfgModule195)

    process.bigCfgModule196 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath196 = cms.Path(process.bigCfgModule196)

    process.bigCfgModule197 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath197 = cms.Path(process.bigCfgModule197)

    process.bigCfgModule198 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath198 = cms.Path(process.bigCfgModule198)

    process.bigCfgModule199 = cms.EDProducer("StreamThingProducer",
                                             array_size = cms.int32(2),
                                             instance_count = cms.int32(2)
                                             )
    process.bigCfgPath199 = cms.Path(process.bigCfgModule199)
