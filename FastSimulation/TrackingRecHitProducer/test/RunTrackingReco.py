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
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source ("PoolSource",
    fileNames=cms.untracked.vstring('file:SimHits.root'),
)

process.recHitProducerSimple=cms.EDProducer("TrackingRecHitProducer",
    simHits = cms.InputTag("famosSimHits","TrackerHits"),
    plugins=cms.VPSet(
        cms.PSet(
            name = cms.string("noSmearing"),
            type=cms.string("TrackingRecHitNoSmearingPlugin"),
            select=cms.string("subdetId==BPX")
        ),
        
        cms.PSet(
            name = cms.string("BPXmonitor"),
            type=cms.string("TrackingRecHitMonitorPlugin"),
            xmax=cms.double(5.0),
            ymax=cms.double(5.0),
            select=cms.string("subdetId==BPX"),

        )
    )
)


process.recHitProducerTemplates=cms.EDProducer("TrackingRecHitProducer",
    simHits = cms.InputTag("famosSimHits","TrackerHits"),
    plugins=cms.VPSet(
        cms.PSet(
            name = cms.string("pixelBarrelSmearer"),
            type=cms.string("PixelBarrelTemplateSmearerPlugin"),
            NewPixelBarrelResolutionFile1 = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrel38T.root'),
            NewPixelBarrelResolutionFile2 = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrelEdge38T.root'),
            NewPixelBarrelResolutionFile3 = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution2014.root'),
            NewPixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionForward38T.root'),
            NewPixelForwardResolutionFile2 = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution2014.root'),
            UseCMSSWPixelParametrization = cms.bool(True),
            probfilebarrel = cms.string('FastSimulation/TrackingRecHitProducer/data/bmergeprob.root'),
            probfileforward = cms.string('FastSimulation/TrackingRecHitProducer/data/fmergeprob.root'),
            templateIdBarrel = cms.int32( 40 ),
            templateIdForward  = cms.int32( 41 ),
            select=cms.string("subdetId==BPX"),
        ),
        
        cms.PSet(
            name = cms.string("BPXmonitor"),
            type=cms.string("TrackingRecHitMonitorPlugin"),
            xmax=cms.double(5.0),
            ymax=cms.double(5.0),
            select=cms.string("subdetId==BPX"),

        )
    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    recHitProducerSimple = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('TRandom3')
    ),
    recHitProducerTemplates = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('TRandom3')
    )
)

process.tracking_step=cms.Path(
#    process.recHitProducerSimple
    process.recHitProducerTemplates
)

process.TFileService = cms.Service("TFileService", fileName = cms.string("histo.root") )

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
