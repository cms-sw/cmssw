import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonConstantsTest")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Geometry.CMSCommonData.cmsExtendedGeometry2026D46XML_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.muonGeometryConstants.fromDD4Hep = False

process.test = cms.EDAnalyzer("MuonGeometryConstantsTester")

process.p1 = cms.Path(process.test)
