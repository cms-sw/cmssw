import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('GeomDump',Phase2C17I13M9)

geomFile = "Geometry.HGCalCommonData.testHGCalPassiveXML_cfi"
fileName = "hgcalPassiveDDD.root"

print("Geometry file: ", geomFile)
print("Output file:   ", fileName)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.HGCalGeom=dict()
#   process.MessageLogger.SimG4CoreGeometry=dict()
#   process.MessageLogger.TGeoMgrFromDdd=dict()
    
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
))

process.dump = cms.EDAnalyzer("DumpSimGeometry",
                              outputFileName = cms.untracked.string(fileName))

process.p = cms.Path(process.dump)
