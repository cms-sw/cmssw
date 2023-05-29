###############################################################################
# Way to use this:
#   cmsRun testHGCalGeometryMouseBite_cfg.py geometry=D88
#
#   Options for type D88, D92, D93, D94
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D92",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: D88, D92, D93, D94")

### get and parse the command line arguments
options.parseArguments()
print(options)

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process("HGCalGeometryMouseBite",Phase2C17I13M9)

####################################################################
# Use the options
if (options.geometry == "D88"):
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
elif (options.geometry == "D93"):
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
elif (options.geometry == "D94"):
    process.load('Configuration.Geometry.GeometryExtended2026D94Reco_cff')
else:
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
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

process.prodEE = cms.EDAnalyzer("HGCalGeometryMouseBiteTester",
                                NameSense     = cms.string("HGCalEESensitive"),
                                NameDevice    = cms.string("HGCal EE"),
)

process.prodHEF = process.prodEE.clone(
    NameSense  = "HGCalHESiliconSensitive",
    NameDevice = "HGCal HE Front"
)

process.prodHFN = process.prodEE.clone(
    NameSense  = "HGCalHFNoseSensitive",
    NameDevice = "HGCal HF Nose"
)

if (options.geometry == "D94"):
    process.p1 = cms.Path(process.generator*process.prodEE*process.prodHEF*process.prodHFN)
else:
    process.p1 = cms.Path(process.generator*process.prodEE*process.prodHEF)
