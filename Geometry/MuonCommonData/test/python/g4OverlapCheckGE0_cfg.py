###############################################################################
# Way to use this:
#   cmsRun g4OverlapCheckGE0_cfg.py geometry=Test tol=0.1
#
#   Options for geometry Test, D99
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "Test",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: Test, D99")
options.register('tol',
                 0.1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Tolerance for checking overlaps: 0.01, 0.1, 1.0"
)

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('OverlapCheck',Phase2C17I13M9)

if (options.geometry == "Test"):
    geomFile = "Geometry.MuonCommonData.testGE0XML_cfi"
else:
    geomFile = "Geometry.MuonCommonData.cmsExtendedGeometry2026D99XML_cfi"

baseName = "cms2026" + options.geometry + "DDD"

process.load(geomFile)
if (options.geometry == "Test"):
    process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
    process.load("Geometry.MuonNumbering.muonOffsetESProducer_cff")
else:
    process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff")
    process.load("SLHCUpgradeSimulations.Geometry.fakePhase2OuterTrackerConditions_cff")
    process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
    process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
    process.load("Geometry.HGCalCommonData.hgcalParametersInitialization_cfi")
    process.load("Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi")
    process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
    process.load("Geometry.MuonNumbering.muonOffsetESProducer_cff")
    process.load("Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff")

process.load('FWCore.MessageService.MessageLogger_cfi')

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)

#if hasattr(process,'MessageLogger'):
#    process.MessageLogger.SimG4CoreGeometry=dict()

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
