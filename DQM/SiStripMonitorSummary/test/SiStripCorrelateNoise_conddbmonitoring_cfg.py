import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("CALIB")

options = VarParsing.VarParsing("analysis")

options.register ('connectionString',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "connection string")
options.register ('noiseTagName',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "noise tag name")
options.register ('gainTagName',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "gain tag name")
options.register ('firstRunNumber',
                  0,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "first run number")
options.register ('secondRunNumber',
                  0,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "second run number")

options.parseArguments()


process.MessageLogger = cms.Service("MessageLogger",
                                    out = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING')),
                                    destinations = cms.untracked.vstring('out','cout')
                                    )

process.source = cms.Source("EmptyIOVSource",
                            firstValue = cms.uint64(options.firstRunNumber),
                            lastValue = cms.uint64(options.secondRunNumber),
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
                                      DBParameters = cms.PSet(messageLevel = cms.untracked.int32(1), # it used to be 2
                                                              authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
                                                              ),
                                      timetype = cms.untracked.string('runnumber'),
                                      connect = cms.string(options.connectionString),
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('SiStripNoisesRcd'),
                                                                 tag = cms.string(options.noiseTagName)
                                                                 ),
                                                        cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                                                 tag = cms.string(options.gainTagName)
                                                                 )
                                                        )
                                      )


process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
process.analysis = cms.EDAnalyzer("SiStripCorrelateNoise")


process.p = cms.Path(process.analysis)

