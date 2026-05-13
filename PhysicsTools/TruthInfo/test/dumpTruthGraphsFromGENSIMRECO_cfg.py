import FWCore.ParameterSet.Config as cms

# user options
import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("inputFile",        nargs='?', default="step3.root",
                    metavar='FILE', help="Input file, default=%(default)r" )
parser.add_argument('-o', "--outdir",   default='',
                    help="output directory, default=%(default)r" )
parser.add_argument('-n', "--maxevts",  type=int, default=-1,
                    help="maximum number of events to process, default=%(default)s" )
parser.add_argument('-m', "--merge",    dest='mergeGenSim', action='store_true',
                    help="merge GEN-SIM nodes of duplicates" )
parser.add_argument('-c', "--collapse", action='store_true', help="collapse GenParticle copies" )
parser.add_argument('-t', "--tag",      default='', help="tag for out put file" )
args = parser.parse_args()
if '/' not in args.inputFile and ':' not in args.inputFile:
    args.inputFile = 'file:'+args.inputFile
if args.outdir and not os.path.exists(args.outdir):
    os.makedirs(args.outdir, exist_ok=True)

process = cms.Process("TRUTHGRAPH")

process.load("FWCore.MessageService.MessageLogger_cfi")

# Needed if TruthLogicalGraphHitIndexProducer does HGCal simId -> reco DetId relabelling.
# Keep this consistent with the geometry used to produce step3.root.
process.load("Configuration.Geometry.GeometryExtendedRun4D120Reco_cff")

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(args.maxevts)
)

process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(
        args.inputFile #"file:step3.root"
    )
)

process.options = cms.untracked.PSet(
    wantSummary=cms.untracked.bool(True)
)

process.truthGraphProducer = cms.EDProducer(
    "TruthGraphProducer",
    genEventHepMC3=cms.InputTag("generatorSmeared"),
    genEventHepMC=cms.InputTag("generatorSmeared"),
    simTracks=cms.InputTag("g4SimHits"),
    simVertices=cms.InputTag("g4SimHits"),
    addGenToSimEdges=cms.bool(True),
)

process.truthGraphDumper = cms.EDAnalyzer(
    "TruthGraphDumper",
    src=cms.InputTag("truthGraphProducer"),
    dotFile=cms.string(os.path.join(args.outdir,f"truthgraph{args.tag}.dot")), # output file
    maxNodes=cms.uint32(20000),
    maxEdgesPerNode=cms.uint32(50),
    simTracks=cms.InputTag("g4SimHits"),
    simVertices=cms.InputTag("g4SimHits"),
    genEventHepMC=cms.InputTag("generatorSmeared"),
    genEventHepMC3=cms.InputTag("generatorSmeared"),
)

process.truthLogicalGraphProducer = cms.EDProducer(
    "TruthLogicalGraphProducer",
    src=cms.InputTag("truthGraphProducer"),
    simTracks=cms.InputTag("g4SimHits"),
    simVertices=cms.InputTag("g4SimHits"),
    genEventHepMC3=cms.InputTag("generatorSmeared"),
    genEventHepMC=cms.InputTag("generatorSmeared"),

    mergeGenSimVertices=cms.bool(True),

    postProcessing=cms.PSet(
        collapseIntermediateGenParticles=cms.bool(True),

        # Empty means: keep the full logical graph.
        # Example: cms.vint32(22) keeps photons as seeds,
        # keeps seedParentDepth parent generations above each seed,
        # then keeps everything downstream.
        seedPdgIds=cms.vint32(23,15,-15,25,4,5,6),

        seedParentDepth=cms.uint32(1),

        # Remove particles by PDG id.
        # Example: cms.vint32(22) removes all photons from the final logical graph.
        ignoredPdgIds=cms.vint32(),

        # Remove particles by logical particle id.
        # These ids refer to the graph after the previous postprocessing steps
        # and before ignored-particle collapsing.
        ignoredParticleIds=cms.vuint32(),
    ),
)
process.simHitToRecHitMapProducer = cms.EDProducer(
    "SimHitToRecHitMapProducer",

    hgcalRecHits=cms.VInputTag(
        cms.InputTag("HGCalRecHit", "HGCEERecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEFRecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEBRecHits", "RECO"),
    ),

    pfRecHits=cms.VInputTag(
        cms.InputTag("particleFlowRecHitECAL", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHBHE", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHF", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHO", "Cleaned", "RECO"),
    ),
)
process.truthLogicalGraphHitIndexProducer = cms.EDProducer(
    "TruthLogicalGraphHitIndexProducer",

    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),

    recHitMap=cms.InputTag("simHitToRecHitMapProducer"),

    simHitCollections=cms.VInputTag(
        cms.InputTag("g4SimHits", "HGCHitsEE", "SIM"),
        cms.InputTag("g4SimHits", "HGCHitsHEfront", "SIM"),
        cms.InputTag("g4SimHits", "HGCHitsHEback", "SIM"),
        cms.InputTag("g4SimHits", "EcalHitsEB", "SIM"),
        cms.InputTag("g4SimHits", "HcalHits", "SIM"),
    ),

    doHGCalRelabelling=cms.bool(False),
)

process.truthLogicalGraphDumper = cms.EDAnalyzer(
    "TruthLogicalGraphDumper",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),

    hgcalRecHits=cms.VInputTag(
        cms.InputTag("HGCalRecHit", "HGCEERecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEFRecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEBRecHits", "RECO"),
    ),

    pfRecHits=cms.VInputTag(
        cms.InputTag("particleFlowRecHitECAL", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHBHE", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHF", "Cleaned", "RECO"),
        cms.InputTag("particleFlowRecHitHO", "Cleaned", "RECO"),
    ),

    dotFile=cms.string(os.path.join(args.outdir,f"truthlogicalgraph{args.tag}.dot")), # output file

    maxParticles=cms.uint32(20000),
    maxVertices=cms.uint32(20000),
    maxEdgesPerNode=cms.uint32(300),

    hideLargeSimSourceVertices=cms.bool(True),
    largeSimSourceVertexMinOutgoing=cms.uint32(50),

    hideZeroSimHitSubgraphs=cms.bool(True),
)


process.load("PhysicsTools.TruthInfo.recHitTable_cff")
# process.load('Configuration.EventContent.EventContent_cff')
process.NANOAODSIMoutput = cms.OutputModule("NanoAODOutputModule",
    compressionAlgorithm = cms.untracked.string('ZLIB'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('NANOAODSIM'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string(os.path.join(args.outdir,f"rechits_nano{args.tag}.root")),
    outputCommands = cms.untracked.vstring(
    'drop *',
    'keep nanoaodFlatTable_*Table_*_*',
    # 'keep edmTriggerResults_*_*_*',
    'keep String_*_genModel_*',
    'keep nanoaodMergeableCounterTable_*Table_*_*',
    'keep nanoaodUniqueString_nanoMetadata_*_*',
    'keep nanoaodFlatTable_*Table*_*_*'
)
)

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.default = cms.untracked.PSet(
    limit=cms.untracked.int32(0)
)
process.MessageLogger.cerr.TruthGraphProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)
process.MessageLogger.cerr.TruthLogicalGraphProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)
process.MessageLogger.cerr.TruthLogicalGraphHitIndexProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)
process.MessageLogger.cerr.SimHitToRecHitMapProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)

process.truthGraph_step = cms.Path(
    process.truthGraphProducer
    + process.truthGraphDumper
    + process.truthLogicalGraphProducer
    + process.simHitToRecHitMapProducer
    + process.truthLogicalGraphHitIndexProducer
    + process.truthLogicalGraphDumper
    + process.recHitTable
)

process.nano_step = cms.EndPath(process.NANOAODSIMoutput)
