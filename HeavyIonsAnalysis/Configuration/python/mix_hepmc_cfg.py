import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")
process.load("HeavyIonsAnalysis.Configuration.EventEmbedding_cff")

process.load("Configuration.EventContent.EventContent_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("HeavyIonsAnalysis.Configuration.HIAnalysisEventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/cmshi/mc/sim/pgun_jpsi2muons_d20080423/pgun_jpsi2muons_d20080423_r000001.root')
)

process.mix = cms.EDFilter("MixingModule",
    process.genEventEmbeddingMixParameters,
    input = cms.SecSource("PoolRASource",
        process.eventEmbeddingSourceParameters,
        fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/cmshi/mc/sim/hydjet_sim_x2_c1_d20080425/hydjet_sim_x2_c1_d20080425_r000002.root')
    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(4),
        VtxSmeared = cms.untracked.uint32(2)
    ),
    sourceSeed = cms.untracked.uint32(1)
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(False),
    ignoreTotal = cms.untracked.int32(0)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('mix'),
    destinations = cms.untracked.vstring('cout', 
        'cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport.xml')
)

process.Timing = cms.Service("Timing")

process.output = cms.OutputModule("PoolOutputModule",
    process.HIRecoObjects,
    compressionLevel = cms.untracked.int32(2),
    commitInterval = cms.untracked.uint32(1),
    fileName = cms.untracked.string('jpsi2muons_PbPb_GEN.root')
)

process.p = cms.Path(process.mix)
process.outpath = cms.EndPath(process.output)
process.output.outputCommands.append('keep *_*_*_GEN')


