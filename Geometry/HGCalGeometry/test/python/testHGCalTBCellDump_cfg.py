import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("HGCalTBCellDump",Phase2C17I13M9)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Geometry.HGCalTBCommonData.testTB181V1XML_cfi")
process.load("Geometry.HGCalTBCommonData.hgcalTBParametersInitialization_cfi")
process.load("Geometry.HGCalTBCommonData.hgcalTBNumberingInitialization_cfi")
process.load("Geometry.CaloTopology.hgcalTBTopologyTester_cfi")
process.load("Geometry.HGCalGeometry.HGCalTBGeometryESProducer_cfi")
process.load("Geometry.HGCalGeometry.hgcalTBGeometryDump_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeomX=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.Timing = cms.Service("Timing")
process.hgcalTBGeometryDump.detectorNames = ["HGCalEESensitive"]

process.p1 = cms.Path(process.hgcalTBGeometryDump)
