# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

import FWCore.ParameterSet.Config as cms

# user options
import os
from argparse import ArgumentParser, BooleanOptionalAction
parser = ArgumentParser()
parser.add_argument("inputFile",        nargs='?', default="step3.root",
                    metavar='FILE', help="Input file, default=%(default)r" )
parser.add_argument('-o', "--outdir",   default='',
                    help="output directory, default=%(default)r" )
parser.add_argument('-n', "--maxevts",  type=int, default=-1,
                    help="maximum number of events to process, default=%(default)s" )
parser.add_argument('-m', "--merge",    dest='mergeGenSim', action=BooleanOptionalAction, default=True,
                    help="merge GEN and SIM vertices (producer-level and by position)" )
parser.add_argument('-c', "--collapse", action=BooleanOptionalAction, default=True,
                    help="collapse intermediate GenParticle copies" )
parser.add_argument('-t', "--tag",      default='', help="tag for out put file" )
parser.add_argument('-s', "--seeds",    default=None,
                    help="comma-separated seed PDG ids, e.g. '15,-15'; '0' keeps the full graph; "
                         "default=%(default)s uses the hardcoded list" )
parser.add_argument('-g', "--groups",   default=None,
                    help="semicolon-separated decay PDG id groups, e.g. '13,-14,16;-13,14,-16'" )
parser.add_argument('-f', "--flavors",   default=None,
                    help="comma-separated heavy-flavor quark ids to seed hadrons on, e.g. '5' for B hadrons, '4' for D" )
parser.add_argument('-d', "--parentDepth", type=int, default=1,
                    help="ancestor generations kept above each root as context, default=%(default)s" )
parser.add_argument('-i', "--ignore",   default=None,
                    help="comma-separated PDG ids to remove from the final logical graph, e.g. '22'" )
parser.add_argument("--keepSpectators", action=BooleanOptionalAction, default=True,
                    help="keep stable final-state spectators (underlying event) outside the selection; "
                         "use --no-keepSpectators for a focused subgraph" )
parser.add_argument("--attachSources", action=BooleanOptionalAction, default=True,
                    help="attach selected roots to artificial Upstream/UnderlyingEvent source vertices; "
                         "use --no-attachSources to root each seed directly (e.g. ten taus -> ten subgraphs)" )
parser.add_argument("--keepProductionSiblings", action=BooleanOptionalAction, default=False,
                    help="also keep the seed's hard-scatter co-products (the other outgoing particles of its "
                         "production vertex and their subtrees), e.g. the VBF tagging quarks/jets recoiling against "
                         "the Higgs - siblings of the seed that seedParentDepth never reaches" )
parser.add_argument("--signal-only", dest='signalOnly', action=BooleanOptionalAction, default=False,
                    help="pile-up filter: keep only the signal interaction (EncodedEventId bx 0, event 0), dropping "
                         "all pile-up; orthogonal to the seed selection" )
parser.add_argument("--bunch-crossings", dest='bunchCrossings', default=None,
                    help="pile-up filter: comma-separated bunch crossings to keep, e.g. '0' for in-time only "
                         "(default: keep all)" )
parser.add_argument("--showAll", action='store_true',
                    help="do not hide zero-simhit subgraphs or large SIM source vertices in the logical DOT dump" )
parser.add_argument("--layout", default="dot",
                    help="DOT layout for the logical-graph dump: 'dot' (default, hierarchical L->R ranks) "
                         "or a force-directed engine ('sfdp'/'fdp'/'neato') for node repulsion + spring edges" )
args = parser.parse_args()

def _parsePdgIds(text):
    return [int(token) for token in text.replace(' ', '').split(',') if token]

seedPdgIds = _parsePdgIds(args.seeds) if args.seeds is not None else [23, 15, -15, 25, 4, 5, 6]
decayPdgIdGroups = [_parsePdgIds(group) for group in args.groups.split(';')] if args.groups else []
ignoredPdgIds = _parsePdgIds(args.ignore) if args.ignore else []
seedHadronFlavors = _parsePdgIds(args.flavors) if args.flavors else []
keepBunchCrossings = _parsePdgIds(args.bunchCrossings) if args.bunchCrossings else []
if '/' not in args.inputFile and ':' not in args.inputFile:
    args.inputFile = 'file:'+args.inputFile
if args.outdir and not os.path.exists(args.outdir):
    os.makedirs(args.outdir, exist_ok=True)

process = cms.Process("TRUTHGRAPH")

process.load("FWCore.MessageService.MessageLogger_cfi")

# Needed if TruthLogicalGraphHitIndexProducer does HGCal simId -> reco DetId relabelling.
# Keep this consistent with the geometry used to produce step3.root.
process.load("Configuration.Geometry.GeometryExtendedRun4D120Reco_cff")

# Use the ideal tracker geometry so the tracker simhit table needs no alignment
# conditions (GlobalPositionRcd) when running standalone without a GlobalTag.
process.trackerGeometry.applyAlignment = cms.bool(False)

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

    mergeGenSimVertices=cms.bool(args.mergeGenSim),

    postProcessing=cms.PSet(
        collapseIntermediateGenParticles=cms.bool(args.collapse),

        # Empty means: keep the full logical graph.
        # The most upstream particle of each matching chain becomes a root and
        # its full downstream subgraph is kept; unselected upstream activity is
        # collapsed into one artificial source vertex.
        # The special value 0 disables the selection and keeps the full graph
        # (debugging escape hatch).
        seedPdgIds=cms.vint32(*seedPdgIds),

        # Seed on hadrons by heavy-flavor content (5=b, 4=c), OR-ed with
        # seedPdgIds. E.g. -f 5 selects all B-hadron decay subgraphs.
        seedHadronFlavors=cms.vint32(*seedHadronFlavors),

        # Ancestor generations kept above each root as context only: their
        # other descendants are not pulled in.
        seedParentDepth=cms.uint32(args.parentDepth),

        # Keep stable spectators (underlying event) on an artificial
        # UnderlyingEvent vertex; --no-keepSpectators drops them for a focused
        # subgraph (only the selection + its Upstream/ISR context).
        keepStableSpectators=cms.bool(args.keepSpectators),

        # Root each selected seed directly (true graph roots) instead of
        # attaching it to an artificial Upstream/UnderlyingEvent vertex.
        # --no-attachSources gives one self-contained subgraph per seed.
        attachSelectionSources=cms.bool(args.attachSources),

        # Also keep the seed's hard-scatter co-products (siblings at its
        # production vertex and their subtrees), e.g. the VBF tagging quarks
        # that become forward jets; --keepProductionSiblings to enable.
        keepProductionSiblings=cms.bool(args.keepProductionSiblings),

        # Pile-up filter (orthogonal to the seed selection): --signal-only keeps
        # only the signal interaction (EncodedEventId bx 0, event 0); a non-empty
        # --bunch-crossings keeps only the listed bunch crossings (e.g. 0 = in-time).
        signalOnly=cms.bool(args.signalOnly),
        keepBunchCrossings=cms.vint32(*keepBunchCrossings),

        # Decay patterns of interest: unordered, charge-sensitive PDG id
        # multisets, OR-ed. With seedPdgIds set, only roots whose effective
        # decay products contain a group are kept, e.g.
        #   cms.PSet(pdgIds=cms.vint32(13, -13))
        # keeps Z -> mu mu but drops Z -> e e. Without seedPdgIds, or when the
        # event contains no seed particle at all, vertices whose outgoing PDG
        # ids contain a group are selected directly.
        decayPdgIdGroups=cms.VPSet(
            *[cms.PSet(pdgIds=cms.vint32(*group)) for group in decayPdgIdGroups]
        ),

        # Remove particles by PDG id.
        # Example: cms.vint32(22) removes all photons from the final logical graph.
        ignoredPdgIds=cms.vint32(*ignoredPdgIds),

        # Remove particles by logical particle id.
        # These ids refer to the graph after the previous postprocessing steps
        # and before ignored-particle collapsing.
        ignoredParticleIds=cms.vuint32(),
    ),
)
process.detIdToRecHitMapProducer = cms.EDProducer(
    "DetIdToRecHitMapProducer",

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

    recHitMap=cms.InputTag("detIdToRecHitMapProducer"),

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

    layout=cms.string(args.layout),

    maxParticles=cms.uint32(20000),
    maxVertices=cms.uint32(20000),
    # --showAll lifts the per-node edge cap: with large events the artificial
    # source vertex legitimately has more than 300 outgoing spectators.
    maxEdgesPerNode=cms.uint32(1000000 if args.showAll else 300),

    hideLargeSimSourceVertices=cms.bool(not args.showAll),
    largeSimSourceVertexMinOutgoing=cms.uint32(50),

    hideZeroSimHitSubgraphs=cms.bool(not args.showAll),
)


process.load("PhysicsTools.TruthInfo.recHitTable_cfi")

# Barrel/forward calorimeter PFRecHits as a separate NanoAOD collection.
# HGCal rechits stay in recHitTable above. NOTE: the offline (RECO) PFRecHit
# collections are empty in this sample, so the HLT-tier PFRecHits (which contain
# the hits) are used here.
process.pfRecHitTable = cms.EDProducer(
    "PFRecHitFlatTableProducer",
    objName=cms.string("pfrechits"),
    label_rechits=cms.VInputTag(
        cms.InputTag("hltParticleFlowRecHitECALUnseeded", "", "HLT"),
        cms.InputTag("hltParticleFlowRecHitHBHE", "", "HLT"),
        cms.InputTag("hltParticleFlowRecHitHF", "", "HLT"),
        cms.InputTag("hltParticleFlowRecHitHO", "", "HLT"),
    ),
)

# Tracker PSimHits as a separate NanoAOD collection.
process.trackerSimHitTable = cms.EDProducer(
    "TrackerSimHitFlatTableProducer",
    objName=cms.string("trackersimhits"),
    label_simhits=cms.VInputTag(
        cms.InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"),
        cms.InputTag("g4SimHits", "TrackerHitsPixelBarrelHighTof"),
        cms.InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"),
        cms.InputTag("g4SimHits", "TrackerHitsPixelEndcapHighTof"),
        cms.InputTag("g4SimHits", "TrackerHitsTIBLowTof"),
        cms.InputTag("g4SimHits", "TrackerHitsTIBHighTof"),
        cms.InputTag("g4SimHits", "TrackerHitsTIDLowTof"),
        cms.InputTag("g4SimHits", "TrackerHitsTIDHighTof"),
        cms.InputTag("g4SimHits", "TrackerHitsTOBLowTof"),
        cms.InputTag("g4SimHits", "TrackerHitsTOBHighTof"),
        cms.InputTag("g4SimHits", "TrackerHitsTECLowTof"),
        cms.InputTag("g4SimHits", "TrackerHitsTECHighTof"),
    ),
)
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
process.MessageLogger.cerr.TruthLogicalGraphPostProcessor = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)
process.MessageLogger.cerr.TruthLogicalGraphHitIndexProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)
process.MessageLogger.cerr.DetIdToRecHitMapProducer = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)

process.truthGraph_step = cms.Path(
    process.truthGraphProducer
    + process.truthGraphDumper
    + process.truthLogicalGraphProducer
    + process.detIdToRecHitMapProducer
    + process.truthLogicalGraphHitIndexProducer
    + process.truthLogicalGraphDumper
    + process.recHitTable
    + process.pfRecHitTable
    + process.trackerSimHitTable
)

process.nano_step = cms.EndPath(process.NANOAODSIMoutput)
