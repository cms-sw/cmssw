import FWCore.ParameterSet.Config as cms

process = cms.Process("makeGlobalPositionRcd")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.GlobalPositionRcdWrite = cms.EDAnalyzer("GlobalPositionRcdWrite",
                                                tracker = cms.PSet(x = cms.double(0.),
                                                                   y = cms.double(0.),
                                                                   z = cms.double(0.),
                                                                   alpha = cms.double(0.),
                                                                   beta = cms.double(0.),
                                                                   gamma = cms.double(0.)),
                                                muon = cms.PSet(x = cms.double(0.),
                                                                y = cms.double(0.),
                                                                z = cms.double(0.),
                                                                alpha = cms.double(0.),
                                                                beta = cms.double(0.),
                                                                gamma = cms.double(0.)),
                                                ecal = cms.PSet(x = cms.double(0.),
                                                                y = cms.double(0.),
                                                                z = cms.double(0.),
                                                                alpha = cms.double(0.),
                                                                beta = cms.double(0.),
                                                                gamma = cms.double(0.)),
                                                hcal = cms.PSet(x = cms.double(0.),
                                                                y = cms.double(0.),
                                                                z = cms.double(0.),
                                                                alpha = cms.double(0.),
                                                                beta = cms.double(0.),
                                                                gamma = cms.double(0.)))

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          connect = cms.string("sqlite_file:inertGlobalPositionRcd.db"),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("inertGlobalPositionRcd"))))

process.Path = cms.Path(process.GlobalPositionRcdWrite)
