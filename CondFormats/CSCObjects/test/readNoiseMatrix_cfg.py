# Read Noise Matrix values - Tim Cox - 04.03.2009
# I intend this to read from the standard cond data files, whatever they are.
# This is for CMSSW_3xx

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_30X::All'
##process.GlobalTag.globaltag = 'IDEAL_V11::All'
process.load("CalibMuon.Configuration.getCSCConditions_frontier_cff")
process.cscConditions.toGet = cms.VPSet(
        cms.PSet(record = cms.string('CSCDBGainsRcd'),
                 tag = cms.string('CSCDBGains_ME42_offline')),
        cms.PSet(record = cms.string('CSCDBNoiseMatrixRcd'),
                 tag = cms.string('CSCDBNoiseMatrix_ME42_offline')),
        cms.PSet(record = cms.string('CSCDBCrosstalkRcd'),
                 tag = cms.string('CSCDBCrosstalk_ME42_offline')),
        cms.PSet(record = cms.string('CSCDBPedestalsRcd'),
                 tag = cms.string('CSCDBPedestals_ME42_offline'))
)
process.es_prefer_cscConditions = cms.ESPrefer("PoolDBESSource","cscConditions")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.analyze = cms.EDAnalyzer("CSCNoiseMatrixDBReadAnalyzer")

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.analyze)
process.ep = cms.EndPath(process.printEventNumber)

