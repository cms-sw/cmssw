import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("SLHCUpgradeSimulations.Geometry.PhaseI_cmsSimIdealGeometryXML_R39F16_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'DESIGN_36_V10::All'

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
