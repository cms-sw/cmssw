import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("GeometryTest", Run3_dd4hep)

process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# process.load('Geometry.CommonTopologies.globalTrackingGeometry_cfi')
# process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['upgrade2021']

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.out = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("DTGeometryAnalyzer",
                              tolerance = cms.untracked.double(1.0e-23)
                             )

process.p1 = cms.Path(process.prod)


