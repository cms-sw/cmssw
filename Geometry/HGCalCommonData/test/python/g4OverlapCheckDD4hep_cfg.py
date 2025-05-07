###############################################################################
# Way to use this:
#   cmsRun g4OverlapCheckDD4hep_cfg.py type=V17 tol=0.01 resol=10000
#
#   Options for type  V17, V19
#               tol   1.0, 0.1, 0.01, 0.0
#               resol 10000, 100000, 1000000
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
                  "type of operations: V17, V19")
options.register('tol',
                 0.01,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Tolerance for checking overlaps: 0.0, 0.01, 0.1, 1.0")
options.register('resol',
                 10000,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Resolution for checking overlaps: 10000, 100000, 1000000")

### get and parse the command line arguments
options.parseArguments()
print(options)

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.Eras.Modifier_phase2_hgcalOnly_cff import phase2_hgcalOnly
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("OverlapCheck",Phase2C17I13M9,dd4hep)

####################################################################
# Use the options
geomFile = "Geometry.HGCalCommonData.testHGCalDD4hep" + options.type + "Reco_cff"
outFile = "hgcal" + options.type + str(options.tol)
resoluton = int(options.resol)

print("Geometry file: ", geomFile)
print("Output file:   ", outFile)
print("Resolution:    ", resoluton)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load(geomFile)

if hasattr(process,'MessageLogger'):
#    process.MessageLogger.SimG4CoreGeometry=dict()
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.EcalGeom=dict()

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)

# enable Geant4 overlap check 
process.g4SimHits.CheckGeometry = True

# Geant4 geometry check 
process.g4SimHits.G4CheckOverlap.OutputBaseName = outFile
process.g4SimHits.G4CheckOverlap.OverlapFlag = True
process.g4SimHits.G4CheckOverlap.Tolerance  = options.tol
process.g4SimHits.G4CheckOverlap.Resolution = int(options.resol)
process.g4SimHits.G4CheckOverlap.Depth      = -1
# tells if NodeName is G4Region or G4PhysicalVolume
process.g4SimHits.G4CheckOverlap.RegionFlag = False
# list of names
process.g4SimHits.G4CheckOverlap.NodeNames  = ['cms:OCMS_1']
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
