import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonConstantsTest")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Geometry.CMSCommonData.cmsExtendedGeometry2026D46XML_cfi")
process.load("Geometry.MuonNumbering.muonDDDConstants_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('Geometry')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.muonDDDConstants.fromDD4Hep = False

process.test = cms.EDAnalyzer("MuonConstantsTester")

process.p1 = cms.Path(process.test)
