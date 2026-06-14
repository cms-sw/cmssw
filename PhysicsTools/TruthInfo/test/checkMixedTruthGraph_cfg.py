# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Phase-A pileup prototype: build the mixed (signal+pileup) raw TruthGraph from
# the MixingModule crossing frames, then the logical graph, then audit topology +
# the per-bunch-crossing (signal vs pileup) breakdown. Input must be a DIGI step
# produced with the SimTrack/SimVertex crossing frames enabled and kept.

import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("inputFile", nargs='?', default="step2_cf.root", metavar='FILE')
parser.add_argument('-n', "--maxevts", type=int, default=3)
args = parser.parse_args()
if '/' not in args.inputFile and ':' not in args.inputFile:
    args.inputFile = 'file:' + args.inputFile

process = cms.Process("MIXTOPO")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(args.maxevts))
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(args.inputFile))
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False))

# The mixed raw graph (signal + pileup) is produced in the DIGI step from the
# transient crossing frames and read back here from the input file.

# Logical graph from the mixed raw graph. SimTrack payload is signal-only
# (g4SimHits); pileup particles keep structure + EncodedEventId but no momentum.
process.truthLogicalGraphProducer = cms.EDProducer(
    "TruthLogicalGraphProducer",
    src=cms.InputTag("truthGraphMixedProducer"),
    simTracks=cms.InputTag("g4SimHits"),
    simVertices=cms.InputTag("g4SimHits"),
    genEventHepMC3=cms.InputTag("generatorSmeared"),
    genEventHepMC=cms.InputTag("generatorSmeared"),
    mergeGenSimVertices=cms.bool(True),
    postProcessing=cms.PSet(
        collapseIntermediateGenParticles=cms.bool(True),
        seedPdgIds=cms.vint32(),
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
    rawSrc=cms.InputTag("truthGraphMixedProducer"),
    src=cms.InputTag("truthLogicalGraphProducer"),
)

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.default = cms.untracked.PSet(limit=cms.untracked.int32(0))
process.MessageLogger.cerr.TruthGraphTopologyChecker = cms.untracked.PSet(limit=cms.untracked.int32(-1))

process.p = cms.Path(process.truthLogicalGraphProducer + process.topoChecker)
