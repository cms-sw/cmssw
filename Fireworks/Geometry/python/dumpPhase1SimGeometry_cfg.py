import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("SLHCUpgradeSimulations.Geometry.Phase1_cmsSimIdealGeometryXML_cff")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
))

process.dump = cms.EDAnalyzer("DumpSimGeometry")

process.p = cms.Path(process.dump)
