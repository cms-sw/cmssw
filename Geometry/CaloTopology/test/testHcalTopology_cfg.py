###############################################################################
# Way to use this:
#   cmsRun testHcalTopology_cfg.py type=2018
#
#   Options for type 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS 
options = VarParsing.VarParsing('standard')
options.register('type',
                 "2018",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024")

### get and parse the command line arguments
options.parseArguments()

geomFile = "Configuration.Geometry.GeometryExtended" + options.type + "Reco_cff"

print(options)
print("Geometry file: ", geomFile)

####################################################################
# Use the options

if (options.type == "2015"):
    from Configuration.Eras.Era_Run2_25ns_cff import Run2_25ns
    process = cms.Process('TestHcalTopology',Run2_25ns)
elif (options.type == "2016"):
    from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
    process = cms.Process('TestHcalTopology',Run2_2016)
elif (options.type == "2017"):
    from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
    process = cms.Process('TestHcalTopology',Run2_2017)
elif (options.type == "2018"):
    from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
    process = cms.Process('TestHcalTopology',Run2_2018)
else:
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process('TestHcalTopology',Run3_DDD)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load(geomFile)
process.load("Geometry.CaloTopology.hcalTopologyTester_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HCalGeom=dict()

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

process.p1 = cms.Path(process.generator*process.hcalTopologyTester)
