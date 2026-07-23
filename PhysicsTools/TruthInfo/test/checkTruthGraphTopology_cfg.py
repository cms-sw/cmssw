# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Audits the raw TruthGraph and the logical truth::Graph for strange topologies
# (many-particle vertices, multi-parent / multi-production particles, cycles,
# disconnected components). Full graph, no seed selection.

import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("inputFile", nargs='?', default="step3.root", metavar='FILE')
parser.add_argument('-n', "--maxevts", type=int, default=5)
# Knobs to isolate the cause of logical-graph pathologies.
parser.add_argument("--collapse", action="store_true", default=True)
parser.add_argument("--no-collapse", dest="collapse", action="store_false")
parser.add_argument("--merge-gensim", action="store_true", default=True)
parser.add_argument("--no-merge-gensim", dest="merge_gensim", action="store_false")
args = parser.parse_args()
if '/' not in args.inputFile and ':' not in args.inputFile:
    args.inputFile = 'file:' + args.inputFile

process = cms.Process("TOPOCHK")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(args.maxevts))
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(args.inputFile))
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False))

process.truthGraphProducer = cms.EDProducer(
    "TruthGraphProducer",
    genEventHepMC3=cms.InputTag("generatorSmeared"),
    genEventHepMC=cms.InputTag("generatorSmeared"),
    simTracks=cms.InputTag("g4SimHits"),
    simVertices=cms.InputTag("g4SimHits"),
    addGenToSimEdges=cms.bool(True),
)

process.truthLogicalGraphProducer = cms.EDProducer(
    "TruthLogicalGraphProducer",
    src=cms.InputTag("truthGraphProducer"),
    simTracks=cms.InputTag("g4SimHits"),
    simVertices=cms.InputTag("g4SimHits"),
    genEventHepMC3=cms.InputTag("generatorSmeared"),
    genEventHepMC=cms.InputTag("generatorSmeared"),
    mergeGenSimVertices=cms.bool(args.merge_gensim),
    postProcessing=cms.PSet(
        collapseIntermediateGenParticles=cms.bool(args.collapse),
        seedPdgIds=cms.vint32(),  # full graph, no selection
        seedHadronFlavors=cms.vint32(),
        seedParentDepth=cms.uint32(0),
        keepStableSpectators=cms.bool(True),
        decayPdgIdGroups=cms.VPSet(),
        ignoredPdgIds=cms.vint32(),
        ignoredParticleIds=cms.vuint32(),
    ),
)

process.topoChecker = cms.EDAnalyzer(
    "TruthGraphTopologyChecker",
    rawSrc=cms.InputTag("truthGraphProducer"),
    src=cms.InputTag("truthLogicalGraphProducer"),
)

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.default = cms.untracked.PSet(limit=cms.untracked.int32(0))
process.MessageLogger.cerr.TruthGraphTopologyChecker = cms.untracked.PSet(limit=cms.untracked.int32(-1))

process.p = cms.Path(process.truthGraphProducer + process.truthLogicalGraphProducer + process.topoChecker)
