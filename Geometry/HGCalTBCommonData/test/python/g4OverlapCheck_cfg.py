###############################################################################
# Way to use this:
#   cmsRun g4OverlapCheck_cfg.py type=TB230FEB tol=0.01
#
#   Options for type TB230FEB, TB230Jul
#               tol 1.0, 0.1, 0.01, 0.0
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "TB230FEB",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: TB230FEB, TB230Jul")
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
geomFile = "Geometry.HGCalTBCommonData.test" + options.type + "XML_cfi"
outFile = "hgcal" + options.type + str(options.tol)

print("Geometry file: ", geomFile)
print("Output file:   ", outFile)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load(geomFile)
process.load('Geometry.HGCalCommonData.hgcalEEParametersInitialization_cfi')
process.load('Geometry.HGCalCommonData.hgcalEENumberingInitialization_cfi')
process.load('Geometry.HcalTestBeamData.hcalTB06Parameters_cff')

if hasattr(process,'MessageLogger'):
#    process.MessageLogger.SimG4CoreGeometry=dict()
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.DDCompactViewImpl=dict()

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
process.g4SimHits.OnlySDs = ['HGCalSensitiveDetector', 'HFNoseSensitiveDetector', 'HGCScintillatorSensitiveDetector', 'HcalTB06BeamDetector']
