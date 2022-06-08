import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C11_cff import Phase2C11

process = cms.Process("OverlapTest",Phase2C11)

process.load('Geometry.HGCalCommonData.testHGCalWaferXML_cfi')

process.load('FWCore.MessageService.MessageLogger_cfi')

#if hasattr(process,'MessageLogger'):
#    process.MessageLogger.SimG4CoreGeometry=dict()

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)

# enable Geant4 overlap check 
process.g4SimHits.CheckGeometry = True
process.g4SimHits.OnlySDs = ['DreamSensitiveDetector']

# Geant4 geometry check 
process.g4SimHits.G4CheckOverlap.OutputBaseName = "hgcalwafer"
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
