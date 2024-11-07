###############################################################################
# Way to use this:
#   cmsRun testHGCalPartialWaferTester_cfg.py geometry=D119
#
#   Options for geometry D95, D96, D98, D99, D100, D101, D102, D103, D104, D105,
#                        D106, D107, D108, D109, D110, D111, D112, D113, D114,
#                        D115, D116
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D110",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D95, D96, D98, D99, D100, D101, D102, D103, D104, D105, D106, D107, D108, D109, D110, D111, D112, D113, D114, D115, D116")

### get and parse the command line arguments
options.parseArguments()
print(options)

####################################################################
# Use the options
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('PartailWaferTester',Phase2C17I13M9)

geomFile = "Configuration.Geometry.GeometryExtendedRun4" + options.geometry + "Reco_cff"

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

process.load("Geometry.HGCalCommonData.hgcalPartialWaferTester_cfi")
process.hgcalPartialWaferTester.nameSense = "HGCalEESensitive"
 
process.p1 = cms.Path(process.generator*process.hgcalPartialWaferTester)
