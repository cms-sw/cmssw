import FWCore.ParameterSet.Config as cms

process = cms.Process("NumberingTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.Geometry.GeometryReco_cff")
process.load("Geometry.CMSCommonData.cmsExtendedGeometryXML_cfi")
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")

#this is always needed if users want access to the vector<GeometricDetExtra>
process.TrackerGeometricDetExtraESModule = cms.ESProducer( "TrackerGeometricDetExtraESModule",
                                                            fromDDD = cms.bool( True )
                                                            )
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.prod = cms.EDAnalyzer("ModuleNumbering")

process.p1 = cms.Path(process.prod)


