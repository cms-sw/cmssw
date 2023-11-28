###############################################################################
# Way to use this:
#   cmsRun g4OverlapCheck_cfg.py type=V17 tol=0.01
#
#   Options for type V16, V17, V17n, V17ng, V18, Wafer, WaferFR, WaferPR
#               tol 1.0, 0.1, 0.01, 0.0
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "V17",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: V16, V17, V17n, V7ng, V18, Wafer, WaferFR, WaferPR")
options.register('tol',
                 0.01,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Tolerance for checking overlaps: 0.0, 0.01, 0.1, 1.0")

### get and parse the command line arguments
options.parseArguments()
print(options)

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("OverlapCheck",Phase2C17I13M9)

####################################################################
# Use the options
geomFile = "Geometry.HGCalCommonData.testHGCal" + options.type + "XML_cfi"
outFile = "hgcal" + options.type + str(options.tol)

print("Geometry file: ", geomFile)
print("Output file:   ", outFile)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load(geomFile)
process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff')
process.load('SLHCUpgradeSimulations.Geometry.fakePhase2OuterTrackerConditions_cff')
process.load('Geometry.EcalCommonData.ecalSimulationParameters_cff')
process.load('Geometry.HcalCommonData.hcalDDDSimConstants_cff')
process.load('Geometry.HGCalCommonData.hgcalParametersInitialization_cfi')
process.load('Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi')
process.load('Geometry.MuonNumbering.muonGeometryConstants_cff')
process.load('Geometry.MuonNumbering.muonOffsetESProducer_cff')
process.load('Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff')

if hasattr(process,'MessageLogger'):
#    process.MessageLogger.SimG4CoreGeometry=dict()
    process.MessageLogger.HGCalGeom=dict()

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)

# enable Geant4 overlap check 
process.g4SimHits.CheckGeometry = True

# Geant4 geometry check 
process.g4SimHits.G4CheckOverlap.OutputBaseName = outFile
process.g4SimHits.G4CheckOverlap.OverlapFlag = True
process.g4SimHits.G4CheckOverlap.Tolerance  = options.tol
process.g4SimHits.G4CheckOverlap.Resolution = 10000
process.g4SimHits.G4CheckOverlap.Depth      = -1
# tells if NodeName is G4Region or G4PhysicalVolume
process.g4SimHits.G4CheckOverlap.RegionFlag = False
# list of names
process.g4SimHits.G4CheckOverlap.NodeNames  = ['OCMS']
# enable dump gdml file 
process.g4SimHits.G4CheckOverlap.gdmlFlag   = False
# if defined a G4PhysicsVolume info is printed
process.g4SimHits.G4CheckOverlap.PVname     = ''
# if defined a list of daughter volumes is printed
process.g4SimHits.G4CheckOverlap.LVname     = ''

# extra output files, created if a name is not empty
process.g4SimHits.FileNameField   = ''
process.g4SimHits.FileNameGDML    = ''
process.g4SimHits.FileNameRegions = ''
#
