import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load('Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhaseIPixel_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.dump = cms.EDAnalyzer("PixelDetIdAnalyzer",
                              level = cms.untracked.int32(1)
                              )

process.p = cms.Path(process.dump)
