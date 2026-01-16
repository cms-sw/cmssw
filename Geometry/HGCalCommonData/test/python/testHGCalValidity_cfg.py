###############################################################################
# Way to use this:
#   cmsRun testHGCalValidityCheck_cfg.py geometry=D121
#
#   Options for geometry D88, D92
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D121",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D88, D92, D110, D121")

### get and parse the command line arguments
options.parseArguments()
print(options)

####################################################################
# Use the options
geomName = "Run4" + options.geometry
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
process = cms.Process('HGCalValidity',ERA)

geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
inFile = "miss" + options.geometry + ".txt"

print("Geometry file: ", geomFile)
print("Input file:    ", inFile)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
#   process.MessageLogger.HGCalGeom=dict()
#   process.MessageLogger.HGCGeom=dict()
    process.MessageLogger.HGCalMiss=dict()

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

process.load("Geometry.HGCalCommonData.hgcalValidityTester_cfi")
process.hgcalValidityTester.fileName = inFile

process.p1 = cms.Path(process.generator*process.hgcalValidityTester)
