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
    input = cms.untracked.int32(100)
)

process.source = cms.Source ("PoolSource",
    fileNames=cms.untracked.vstring('file:SimHits.root'),
)

trackerStripGaussianResolutions={
    "TIB": {
        1: cms.double(0.00195),
        2: cms.double(0.00191),
        3: cms.double(0.00325),
        4: cms.double(0.00323)
    },
    "TID": {
        1: cms.double(0.00262),
        2: cms.double(0.00354),
        3: cms.double(0.00391)
    },
    "TOB": {
        1: cms.double(0.00461),
        2: cms.double(0.00458),
        3: cms.double(0.00488),
        4: cms.double(0.00491),
        5: cms.double(0.00293),
        6: cms.double(0.00299)
    },
    "TEC": {
        1: cms.double(0.00262),
        2: cms.double(0.00354),
        3: cms.double(0.00391),
        4: cms.double(0.00346),
        5: cms.double(0.00378),
        6: cms.double(0.00508),
        7: cms.double(0.00422),
    }  
}

process.recHitProducerTemplates=cms.EDProducer("TrackingRecHitProducer",
    simHits = cms.InputTag("famosSimHits","TrackerHits"),
    plugins=cms.VPSet(
    )
)

for subdetId,trackerLayers in trackerStripGaussianResolutions.iteritems():
    for trackerLayer, resolutionX in trackerLayers.iteritems():
        pluginConfig = cms.PSet(
            name = cms.string(subdetId+str(trackerLayer)),
            type=cms.string("TrackingRecGaussianSmearingPlugin"),
            resolutionX=resolutionX,
            select=cms.string("(subdetId=="+subdetId+") && (layer=="+str(trackerLayer)+")"),
        )
        process.recHitProducerTemplates.plugins.append(pluginConfig)
        
for subdetId in ["BPX","FPX","TIB","TID","TOB","TEC"]:
    pluginConfig = cms.PSet(
        name = cms.string("monitor"+subdetId),
        type=cms.string("TrackingRecHitMonitorPlugin"),
        dxmax=cms.double(0.05),
        dymax=cms.double(20.0),
        select=cms.string("subdetId=="+subdetId),
    )
    process.recHitProducerTemplates.plugins.append(pluginConfig)

    

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

