###############################################################################
# Way to use this:
#   cmsRun runHGCGeomCheck_cfg.py type=EE
#
#   Options for type EE, HEsil, HEmix
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "EE",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: EE, HEsil, HEmix")

### get and parse the command line arguments
options.parseArguments()
print(options)

from Configuration.Eras.Era_Phase2C11_cff import Phase2C11

process = cms.Process("OverlapTest",Phase2C11)

####################################################################
# Use the options
if (options.type == "EE"):
    process.load('Geometry.HGCalCommonData.testHGCalEEV17XML_cfi')
    outFile = 'hgcalEE17'
elif (options.type == "HEsil"):
    process.load('Geometry.HGCalCommonData.testHGCalHEsilV17XML_cfi')
    outFile = 'hgcalHEsil17'
else:
    process.load('Geometry.HGCalCommonData.testHGCalHEmixV17XML_cfi')
    outFile = 'hgcalHEmix17'

print("Output file: ", outFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
#   process.MessageLogger.SimG4CoreGeometry=dict()
    process.MessageLogger.HGCalGeom=dict()

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)

# enable Geant4 overlap check 
process.g4SimHits.CheckGeometry = True
process.g4SimHits.OnlySDs = ['DreamSensitiveDetector']

# Geant4 geometry check 
process.g4SimHits.G4CheckOverlap.OutputBaseName = outFile
process.g4SimHits.G4CheckOverlap.OverlapFlag = True
process.g4SimHits.G4CheckOverlap.Tolerance  = 0.01
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
