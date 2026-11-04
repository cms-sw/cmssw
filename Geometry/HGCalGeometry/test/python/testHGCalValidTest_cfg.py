###############################################################################
# Way to use this:
#   cmsRun testHGCalValidTest_cfg.py geometry=D120 detector=HGCalEESensitive
#
#   Options for geometry D120, D122
#           for detector HGCalHEScintillatorSensitive, HGCalEESensitive,
#                        HGCalHESiliconSensitive
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
                  "geometry of operations: D120, D121, D122")
options.register('detector',
                 "HGCalEESensitive",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: HGCalHEScintillatorSensitive, HGCalEESensitive, HGCalHESiliconSensitive")
### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

geomName = "Run4" + options.geometry
detector = options.detector
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)

print("Geometry Name:   ", geomName)
print("Geom file Name:  ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)
print("Detector Name:   ", detector)

process = cms.Process('PROD',ERA)

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

process.load("Geometry.HGCalGeometry.hgcalValidTestHEB_cfi")

process.hgcalValidTestHEF = process.hgcalValidTestHEB.clone(
    detector   = "HGCalHESiliconSensitive",
)

process.hgcalValidTestEE = process.hgcalValidTestHEB.clone(
    detector   = "HGCalEESensitive"
)

if   detector == "HGCalEESensitive":
    process.p1 = cms.Path(process.generator*process.hgcalValidTestEE)
elif detector == "HGCalHESiliconSensitive":
    process.p1 = cms.Path(process.generator*process.hgcalValidTestHEF)
elif detector == "HGCalHEScintillatorSensitive":
    process.p1 = cms.Path(process.generator*process.hgcalValidTestHEB)
else:
    process.p1 = cms.Path(process.generator*process.hgcalValidTestEE*process.hgcalValidTestHEF*process.hgcalValidTestHEB)
