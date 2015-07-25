import FWCore.ParameterSet.Config as cms

process = cms.Process('TRACKRECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('FastSimulation.Configuration.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('FastSimulation.Configuration.Geometries_MC_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedNominalCollision2015_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('FastSimulation.Configuration.Reconstruction_BefMix_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source ("PoolSource",
    fileNames=cms.untracked.vstring('file:SimHits.root'),
)

process.recHitProducer=cms.EDProducer("TrackingRecHitProducer",
    simHits = cms.InputTag("famosSimHits","TrackerHits"),

    plugins=cms.PSet(
        defaultPlugin = cms.PSet(
            type=cms.string("PixelBarrelTemplateSmearerPlugin"),
            #select=cms.string("true || (tecGlued && (subdetId==!FPX)) || ((BPX==subdetId) && (layer!=2) && (pxbModule>5))"),
            select=cms.string("tibGlued || tidGlued || tobGlued || tecGlued"),

        )
    )
)

process.RandomNumberGeneratorService. recHitProducer = cms.PSet(
    initialSeed = cms.untracked.uint32(12345),
    engineName = cms.untracked.string('TRandom3')
  )

process.tracking_step=cms.Path(
    process.siTrackerGaussianSmearingRecHits
    *process.recHitProducer
    #*process.iterTracking
)

process.output = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('tracking_step')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('TRACKRECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(1048576),
    fileName = cms.untracked.string('RecHits.root'),
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *'
    ),
    splitLevel = cms.untracked.int32(0)
)

process.output_step = cms.EndPath(process.output)
