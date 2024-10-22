###############################################################################
# Way to use this:
#   cmsRun testHGCalParametersDDD_cfg.py type=V17
#
#   Options for type V16, V17, V17n, V17Shift, V18
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "V17",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: V16, V17, V17n, V17Shift, V18")

### get and parse the command line arguments
options.parseArguments()
print(options)

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("HGCalParametersTest",Phase2C17I13M9)

geomFile = "Geometry.HGCalCommonData.testHGCal" + options.type + "XML_cfi"

print("Geometry file: ", geomFile)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Geometry.HGCalCommonData.hgcalParametersInitialization_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

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
 
process.load("Geometry.HGCalCommonData.hgcParameterTesterEE_cfi")

process.hgcParameterTesterHESil = process.hgcParameterTesterEE.clone(
    Name = cms.string("HGCalHESiliconSensitive")
)

process.hgcParameterTesterHESci = process.hgcParameterTesterEE.clone(
    Name = cms.string("HGCalHEScintillatorSensitive"),
    Mode = cms.int32(2)
)
 
process.p1 = cms.Path(process.generator*process.hgcParameterTesterEE*process.hgcParameterTesterHESil*process.hgcParameterTesterHESci)
