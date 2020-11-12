import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load('GeneratorInterface.Core.genFilterSummary_cff')

process.load("Geometry.MuonCommonData.testMFXML_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load("Geometry.HGCalCommonData.hgcalParametersInitialization_cfi")
process.load("Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi")

process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run1_mc']

process.load('FWCore.MessageService.MessageLogger_cfi')
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('G4cerr')
    process.MessageLogger.categories.append('G4cout')
    process.MessageLogger.categories.append('MuonGeom')

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(1.6),
        MaxEta = cms.double(2.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(99.99),
        MaxE   = cms.double(100.01)
    ),
    AddAntiParticle = cms.bool(False),
    firstRun        = cms.untracked.uint32(1),
    Verbosity       = cms.untracked.int32(0)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)

process.schedule = cms.Schedule(process.generation_step,
                                process.genfiltersummary_step,
                                process.simulation_step)
process.g4SimHits.UseMagneticField = False

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

