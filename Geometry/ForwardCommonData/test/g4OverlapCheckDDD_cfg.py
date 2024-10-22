###############################################################################
# Way to use this:
#   cmsRun g4OverlapCheckDDD_cfg.py geometry=2021 tol=0.01
#
#   Options for geometry 2021, 2023
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "2023",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: 2021, 2023")
options.register('tol',
                 0.01,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Tolerance for checking overlaps: 0.0, 0.01, 0.1, 1.0"
)

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

baseName = "cmsDDD" + options.geometry
geomName = "Geometry.ForwardCommonData.cmsExtendedGeometry" + options.geometry + "XML_cfi"

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
process = cms.Process("G4PrintGeometry",Run3_DDD)

print("Base file Name: ", baseName)
print("Geom file Name: ", geomName)

process.load("Geometry.ForwardCommonData.testForwardXML_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load(geomName)
process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff')
process.load('Geometry.EcalCommonData.ecalSimulationParameters_cff')
process.load('Geometry.HcalCommonData.hcalDDDSimConstants_cff')
process.load('Geometry.MuonNumbering.muonGeometryConstants_cff')
process.load('Geometry.MuonNumbering.muonOffsetESProducer_cff')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cout=dict()
#   process.MessageLogger.ForwardGeom=dict()

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)

# enable Geant4 overlap check 
process.g4SimHits.CheckGeometry = True

# Geant4 geometry check 
process.g4SimHits.G4CheckOverlap.OutputBaseName = cms.string(baseName)
process.g4SimHits.G4CheckOverlap.OverlapFlag = cms.bool(True)
process.g4SimHits.G4CheckOverlap.Tolerance  = cms.double(options.tol)
process.g4SimHits.G4CheckOverlap.Resolution = cms.int32(10000)
process.g4SimHits.G4CheckOverlap.Depth      = cms.int32(-1)
# tells if NodeName is G4Region or G4PhysicalVolume
process.g4SimHits.G4CheckOverlap.RegionFlag = cms.bool(False)
# list of names
process.g4SimHits.G4CheckOverlap.NodeNames  = cms.vstring('OCMS')
# enable dump gdml file 
process.g4SimHits.G4CheckOverlap.gdmlFlag   = cms.bool(False)
# if defined a G4PhysicsVolume info is printed
process.g4SimHits.G4CheckOverlap.PVname     = ''
# if defined a list of daughter volumes is printed
process.g4SimHits.G4CheckOverlap.LVname     = ''

# extra output files, created if a name is not empty
process.g4SimHits.FileNameField   = ''
process.g4SimHits.FileNameGDML    = ''
process.g4SimHits.FileNameRegions = ''
#
process.g4SimHits.OnlySDs = cms.vstring('ZdcSensitiveDetector', 'TotemT2ScintSensitiveDetector', 'TotemSensitiveDetector', 'RomanPotSensitiveDetector', 'PLTSensitiveDetector', 'MuonSensitiveDetector', 'MtdSensitiveDetector', 'BCM1FSensitiveDetector', 'EcalSensitiveDetector', 'CTPPSSensitiveDetector', 'BSCSensitiveDetector', 'CTPPSDiamondSensitiveDetector', 'FP420SensitiveDetector', 'BHMSensitiveDetector', 'CastorSensitiveDetector', 'CaloTrkProcessing', 'HcalSensitiveDetector', 'TkAccumulatingSensitiveDetector')
