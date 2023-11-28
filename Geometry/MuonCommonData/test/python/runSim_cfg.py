import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD

process = cms.Process("PROD",Run3_DDD)
process.load("SimG4CMS.Calo.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Geometry.MuonCommonData.GeometryExtended2021Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13p6TeVEarly2022Collision_cfi')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2023_realistic', '')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(0.1309),
        MaxEta = cms.double(0.1309),
        MinPhi = cms.double(-3.1415926),
        MaxPhi = cms.double(3.1415926),
        MinE  = cms.double(50.),
        MaxE  = cms.double(50.)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False),
    firstRun        = cms.untracked.uint32(1)
)

process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('simeventMuon.root')
)

process.Timing = cms.Service("Timing")

#process.Tracer = cms.Service("Tracer")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('runMuon.root')
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.analysis_step = cms.EndPath(process.CaloSimHitStudy)
process.out_step = cms.EndPath(process.output)

process.g4SimHits.LHCTransport = False
process.g4SimHits.G4Commands = ['/tracking/verbose 1']

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.genfiltersummary_step,
                                process.simulation_step,
                                process.endjob_step,
                                process.analysis_step,
                                process.out_step
                                )
# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
