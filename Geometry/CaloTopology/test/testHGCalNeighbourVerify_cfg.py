###############################################################################
# Way to use this:
#   cmsRun testHGCalNeighbour_cfg.py geometry=D120 detector=HGCalEESensitive
#                                    waferU=2 waderV=0 cellU=10 cellV=0
#
#   Options for geometry D120, D122
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
                 "D120",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D120, D122")
options.register('detector',
                 "HGCalEESensitive",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: HGCalEESensitive, HGCalHESiliconSensitive")
options.register('waferU',
                 2,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float)
options.register('waferV',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float)
options.register('cellU',
                 10,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float)
options.register('cellV',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float)

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

geomName = "Run4" + options.geometry
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
detector = options.detector
waferU   = int(options.waferU)
waferV   = int(options.waferV)
cellU    = int(options.cellU)
cellV    = int(options.cellV)
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
print("Geometry file: ", geomFile)
print("Detector:      ", detector)
print("WaferU:        ", waferU)
print("WaferV:        ", waferV)
print("CellU:         ", cellU)
print("CellV:         ", cellV)

process = cms.Process('HGCNeighbour',ERA)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Geometry.CaloTopology.hgcalNeighbourVerify_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, GLOBAL_TAG, '')

if hasattr(process,'MessageLogger'):
#   process.MessageLogger.HGCGeom=dict()
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

process.hgcalNeighbourVerify.nameDetector = detector
process.hgcalNeighbourVerify.waferU       = waferU
process.hgcalNeighbourVerify.waferV       = waferV
process.hgcalNeighbourVerify.cellU        = cellU
process.hgcalNeighbourVerify.cellV        = cellV

process.p1 = cms.Path(process.generator*process.hgcalNeighbourVerify)
