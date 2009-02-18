# Read Noise Matrix values - Tim Cox - 18.02.2009
# I intend this to read from the standard cond data files, whatever they are.
# This is for CMSSW_3xx

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_30X::All'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.analyze = cms.EDAnalyzer("CSCNoiseMatrixDBReadAnalyzer")

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.analyze)
process.ep = cms.EndPath(process.printEventNumber)

