import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonNumbering")

process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MuonNumberingTester = cms.EDAnalyzer("MuonNumberingTester")
process.DTNumberingTester = cms.EDAnalyzer("DTNumberingTester")
process.CSCNumberingTester = cms.EDAnalyzer("CSCNumberingTester")
process.RPCNumberingTester = cms.EDAnalyzer("RPCNumberingTester")

process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring('MuonNumberingTester',
                                                                         'DTNumberingTester',
                                                                         'CSCNumberingTester',
                                                                         'RPCNumberingTester'),
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG'),
                                                              noLineBreaks = cms.untracked.bool(True)
                                                              )
                                    )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.p = cms.Path(process.MuonNumberingTester+process.DTNumberingTester+process.CSCNumberingTester+process.RPCNumberingTester)
