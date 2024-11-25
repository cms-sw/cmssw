import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process("ME0GeometryWriter",Phase2C17I13M9)
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D110_cff')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load('Configuration.StandardSequences.DD4hep_GeometrySimPhase2_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('myLog'),
    myLog = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
    )
)

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.ME0GeometryWriter = cms.EDAnalyzer("ME0RecoIdealDBLoader")

process.CondDB.timetype = cms.untracked.string('runnumber')
process.CondDB.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('ME0RecoGeometryRcd'),tag = cms.string('ME0RECO_Geometry_Test01'))
                                                            )
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.ME0GeometryWriter)
