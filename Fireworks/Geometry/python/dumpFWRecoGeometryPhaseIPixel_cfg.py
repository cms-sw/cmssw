import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")
process.load('Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhaseIPixel_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')


process.add_(cms.ESProducer("FWRecoGeometryESProducer"))

#Adding Timing service:
process.Timing = cms.Service("Timing")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.dump = cms.EDAnalyzer("DumpFWRecoGeometry",
                              level = cms.untracked.int32(1)
                              )

process.p = cms.Path(process.dump)
