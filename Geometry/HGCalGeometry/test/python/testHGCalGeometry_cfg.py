###############################################################################
# Way to use this:
#   cmsRun testHGCalGeometry_cfg.py geometry=V17
#
#   Options for geometry V16, V17, V18, V19
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "V17",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: V16, V17, V18, V19")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('HGCalGeometry',Phase2C17I13M9)

geomFile = "Geometry.HGCalCommonData.testHGCal" + options.geometry + "XML_cfi"
print("Geometry file: ", geomFile)

process.load(geomFile)
process.load("Geometry.HGCalCommonData.hgcalParametersInitialization_cfi")
process.load("Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Geometry.CaloEventSetup.HGCalTopology_cfi")
process.load("Geometry.HGCalGeometry.HGCalGeometryESProducer_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.HGCalGeomX=dict()

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

process.load("Geometry.HGCalGeometry.hgcalGeometryTesterEE_cfi")

process.hgcalGeometryTesterHEF = process.hgcalGeometryTesterEE.clone(
    Detector   = "HGCalHESiliconSensitive",
)

process.hgcalGeometryTesterHEB = process.hgcalGeometryTesterEE.clone(
    Detector   = "HGCalHEScintillatorSensitive",
)

#process.p1 = cms.Path(process.generator*process.hgcalGeometryTesterEE*process.hgcalGeometryTesterHEF*process.hgcalGeometryTesterHEB)
process.p1 = cms.Path(process.generator*process.hgcalGeometryTesterEE*process.hgcalGeometryTesterHEF)
#process.p1 = cms.Path(process.generator*process.hgcalGeometryTesterHEB)
