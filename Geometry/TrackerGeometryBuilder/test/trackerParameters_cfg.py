import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.autoCond import autoCond

process = cms.Process("TrackerParametersTest")
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = autoCond['run1_mc']

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.test = cms.EDAnalyzer("TrackerParametersAnalyzer")

process.p1 = cms.Path(process.test)



