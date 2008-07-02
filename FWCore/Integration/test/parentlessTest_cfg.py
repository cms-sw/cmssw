import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.cmsExceptionsFatal_cff")

# The following two lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []
process.MessageLogger.fwkJobReports = []

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.source = cms.Source("EmptySource")

process.ints = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2112)
)

process.maker = cms.EDProducer("ProdigalProducer",
    label = cms.string('ints')
)

process.analyzer = cms.EDAnalyzer("ProdigalAnalyzer")

process.p = cms.Path(process.ints * process.maker * process.analyzer)
