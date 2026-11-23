###############################################################################
# Geant4 overlap check of the BTL test beam 2025 (DUT and REF) geometry (DD4hep)
#   cmsRun g4OverlapCheck_btltb2025_cfg.py tol=0.0
# Adapted from SimG4Core/PrintGeomInfo/test/python/g4OverlapCheckDD4hep_cfg.py
###############################################################################
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('standard')
options.register('tol',
                 0.0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Tolerance for checking overlaps: 0.0, 0.01, 0.1, 1.0"
)
options.parseArguments()

baseName = "btltb2025DD4hep"

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
process = cms.Process('G4PrintGeometry', Phase2C22I13M9, dd4hep)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Geometry.MTDTBCommonData.GeometryDD4hepExtendedBTLTB2025Reco_DUTbehind_cff')

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)
# only the MTD sensitive detector is present in the test beam geometry
process.g4SimHits.OnlySDs = ['MtdSensitiveDetector']

# enable Geant4 overlap check
process.g4SimHits.CheckGeometry = True

# Geant4 geometry check
process.g4SimHits.G4CheckOverlap.NodeName = ""
process.g4SimHits.G4CheckOverlap.OutputBaseName = baseName
process.g4SimHits.G4CheckOverlap.OverlapFlag = True
process.g4SimHits.G4CheckOverlap.Tolerance  = options.tol
process.g4SimHits.G4CheckOverlap.Resolution = 10000
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
