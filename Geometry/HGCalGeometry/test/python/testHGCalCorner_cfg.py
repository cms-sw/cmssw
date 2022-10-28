###############################################################################
# Way to use this:
#   cmsRun testHGCalCorner_cfg.py geometry=D88
#
#   Options for type D88, D92, D93
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
                  "type of operations: D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()
print(options)

from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9

process = cms.Process("HGCalCornerTest",Phase2C11I13M9)

####################################################################
# Use the options
if (options.geometry == "D88"):
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
elif (options.geometry == "D93"):
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
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

process.prodEE = cms.EDAnalyzer("HGCalGeometryCornerTester",
                                detector   = cms.string("HGCalEESensitive"),
                                cornerType = cms.int32(0)
                                )

process.prodHEF = process.prodEE.clone(
    detector   = "HGCalHESiliconSensitive",
)

process.prodHEB = process.prodEE.clone(
    detector   = "HGCalHEScintillatorSensitive",
)

process.p1 = cms.Path(process.generator*process.prodEE*process.prodHEF)
#process.p1 = cms.Path(process.prodHEB)
#process.p1 = cms.Path(process.generator*process.prodEE*process.prodHEF*process.prodHEB)
