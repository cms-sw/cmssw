###############################################################################
# Way to use this:
#   cmsRun testHGCalWafer_cfg.py geometry=D88
#
#   Options for geometry D88, D92, D93
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D88",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('HGCalWafer',Phase2C17I13M9)

geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "_cff"

print("Geometry file: ", geomFile)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
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

process.load("Geometry.HGCalCommonData.hgcalWaferTesterEE_cfi")

process.hgcalWaferTesterHEF = process.hgcalWaferTesterEE.clone(
    NameSense  = "HGCalHESiliconSensitive",
    NameDevice = "HGCal HE Front",
)
 
 
process.p1 = cms.Path(process.generator*process.hgcalWaferTesterEE*process.hgcalWaferTesterHEF)
