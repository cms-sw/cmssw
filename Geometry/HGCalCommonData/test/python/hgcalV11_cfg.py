import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C9_cff import Phase2C9

process = cms.Process('SIM',Phase2C9)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
#process.load("Configuration.Geometry.GeometryExtended2026D49_cff")
process.load("Geometry.HGCalCommonData.testHGCalV11XML_cfi")
process.load("Geometry.HGCalCommonData.hgcalParametersInitialization_cfi")
process.load("Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase2_realistic']

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.HGCSim=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.Timing = cms.Service("Timing")
process.source = cms.Source("EmptySource",
                            firstRun        = cms.untracked.uint32(1),
                            firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
                                   PGunParameters = cms.PSet(
                                       PartID = cms.vint32(11),
                                       MinEta = cms.double(1.60),
                                       MaxEta = cms.double(2.95),
                                       MinPhi = cms.double(-3.14159265359),
                                       MaxPhi = cms.double(3.14159265359),
                                       MinE   = cms.double(99.99),
                                       MaxE   = cms.double(100.01)
                                   ),
                                   AddAntiParticle = cms.bool(True),
                                   Verbosity       = cms.untracked.int32(0),
                               )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simevent.root')
)

process.ProductionFilterSequence = cms.Sequence(process.generator)
process.g4SimHits.HGCSD.CornerMinMask = 3

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.out_step
                                )

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
