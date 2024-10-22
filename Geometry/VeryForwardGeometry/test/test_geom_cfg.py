import FWCore.ParameterSet.Config as cms

process = cms.Process("CTPPS")

from FWCore.MessageLogger.MessageLogger_cfi import *

process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

from Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi import XMLIdealGeometryESSource_CTPPS

process.XMLIdealGeometryESSource = XMLIdealGeometryESSource_CTPPS.clone()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.prod = cms.EDProducer("GeometryProducer",
    MagneticField = cms.PSet(
        delta = cms.double(1.0)
    ),
    UseMagneticField = cms.bool(False),
    UseSensitiveDetectors = cms.bool(False)
)

process.add_(
    cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
    )
)

process.dump = cms.EDAnalyzer("DumpSimGeometry",
    outputFileName = cms.untracked.string('ctppsGeometry.root')
)

process.p = cms.Path(process.prod+process.dump)
