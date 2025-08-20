import FWCore.ParameterSet.Config as cms

geomName = "Run4D110"
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
print("Geometry Name:   ", geomName)
print("Geom file Name:  ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

process = cms.Process("CaloCellGeometryTest",ERA)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.HGCalGeomX=dict()
    process.MessageLogger.CaloGeometryBuilder=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.gtest = cms.EDAnalyzer("CaloCellGeometryTester")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.gtest)
