###############################################################################
# Way to use this:
#   cmsRun testHGCalBadIdCheck_cfg.py geometry=D122 detector=HGCalEESensitive
#                                     fileName=D122FE.txt
#
#   Options for geometry D120, D122
#           for fileName D120FE.txt, D120FH.txt, D122FE.txt, D122FH.txt
#           for detector HGCalEESensitive, HGCalHESiliconSensitive
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D122",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D120, D122")
options.register('fileName',
                 "D122FE.txt",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D120FE.txt, D120FH.txt, D122FE.txt, D122FH.txt")
options.register('detector',
                 "HGCalEESensitive",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: HGCalEESensitive, HGCalHESiliconSensitive")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

geomName = "Run4" + options.geometry
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
detector = options.detector
fileName = options.fileName
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
print("Geometry file: ", geomFile)
print("Input file:    ", fileName)
print("Deector:       ", detector)

process = cms.Process('HGCBadIdCheck',ERA)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Geometry.HGCalGeometry.hgcalBadIdCheck_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, GLOBAL_TAG, '')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCGeom=dict()
    process.MessageLogger.HGCalGeom=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(14),
        MinEta = cms.double(-3.5),
        MaxEta = cms.double(3.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(9.99),
        MaxE   = cms.double(10.01)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.hgcalBadIdCheck.nameDetector = detector
process.hgcalBadIdCheck.fileName = fileName

process.p1 = cms.Path(process.generator*process.hgcalBadIdCheck)
