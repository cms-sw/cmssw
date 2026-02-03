###############################################################################
# Way to use this:
#   cmsRun testHGCalIdCheck_cfg.py geometry=D120 detector=HGCalEESensitive
#                                  fileIn=D120E fileOut=junk mode=0 cog=10
#
#   Options for geometry D120, D122
#           for fileIn D120E.txt, D120H.txt D122E.txt, D122H.txt, ""
#           for fileOut D120E.out, D120H.out, D122E.out, D122H.out, ""
#           for detector HGCalEESensitive, HGCalHESiliconSensitive
#           for outMode 0, 1
#           for cog 0, 10
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D120",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D120, D122")
options.register('fileIn',
                 "D120E.txt",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Input File name: D120E.txt, D120H.txt, D122E.txt, D122H.txt")
options.register('fileOut',
                 "",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Outout File name: D120E.out, D120H.out, D122E.out, D122H.out")
options.register('detector',
                 "HGCalEESensitive",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "detector name: HGCalEESensitive, HGCalHESiliconSensitive")
options.register('outMode',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Output mode: 0, 1")
options.register('cog',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Use of cell center: 0, 10")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

geomName = "Run4" + options.geometry
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
detector = options.detector
fileName = options.fileIn
outFile  = options.fileOut
outMode  = int(options.outMode)
cog      = int(options.cog)
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
print("Geometry file: ", geomFile)
print("Input file:    ", fileName)
print("Output file:   ", outFile)
print("Detector:      ", detector)
print("OutMode:       ", outMode)
print("COG:           ", cog)

process = cms.Process('HGCIdCheck',ERA)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Geometry.HGCalGeometry.hgcalIdCheck_cfi')
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

process.hgcalIdCheck.nameDetector = detector
process.hgcalIdCheck.fileName = fileName
process.hgcalIdCheck.outFileName = outFile
process.hgcalIdCheck.mode = outMode
process.hgcalIdCheck.cog  = cog

process.p1 = cms.Path(process.generator*process.hgcalIdCheck)
