# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("inputFile", nargs='?', default="step3.root", metavar='FILE')
parser.add_argument('-n', "--maxevts", type=int, default=3)
args = parser.parse_args()
if '/' not in args.inputFile and ':' not in args.inputFile:
    args.inputFile = 'file:' + args.inputFile

process = cms.Process("BRANCHVAL")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryExtendedRun4D120Reco_cff")
process.trackerGeometry.applyAlignment = cms.bool(False)

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
    mergeGenSimVertices=cms.bool(True),
    postProcessing=cms.PSet(
        collapseIntermediateGenParticles=cms.bool(True),
        seedPdgIds=cms.vint32(),  # full graph, no selection
        seedHadronFlavors=cms.vint32(),
        seedParentDepth=cms.uint32(0),
        keepStableSpectators=cms.bool(True),
        decayPdgIdGroups=cms.VPSet(),
        ignoredPdgIds=cms.vint32(),
        ignoredParticleIds=cms.vuint32(),
    ),
)

process.detIdToRecHitMapProducer = cms.EDProducer(
    "DetIdToRecHitMapProducer",
    hgcalRecHits=cms.VInputTag(
        cms.InputTag("HGCalRecHit", "HGCEERecHits"),
        cms.InputTag("HGCalRecHit", "HGCHEFRecHits"),
        cms.InputTag("HGCalRecHit", "HGCHEBRecHits"),
    ),
    pfRecHits=cms.VInputTag(
        cms.InputTag("particleFlowRecHitECAL", "Cleaned"),
        cms.InputTag("particleFlowRecHitHBHE", "Cleaned"),
        cms.InputTag("particleFlowRecHitHF", "Cleaned"),
        cms.InputTag("particleFlowRecHitHO", "Cleaned"),
    ),
)

process.truthLogicalGraphHitIndexProducer = cms.EDProducer(
    "TruthLogicalGraphHitIndexProducer",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    recHitMap=cms.InputTag("detIdToRecHitMapProducer"),
    simHitCollections=cms.VInputTag(
        cms.InputTag("g4SimHits", "HGCHitsEE"),
        cms.InputTag("g4SimHits", "HGCHitsHEfront"),
        cms.InputTag("g4SimHits", "HGCHitsHEback"),
        cms.InputTag("g4SimHits", "EcalHitsEB"),
        cms.InputTag("g4SimHits", "HcalHits"),
    ),
    doHGCalRelabelling=cms.bool(False),
)

process.branchValidator = cms.EDAnalyzer(
    "BranchTruthReplacementValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    caloParticles=cms.InputTag("mix", "MergedCaloTruth"),
    simClusters=cms.InputTag("mix", "MergedCaloTruth"),
)

# ClusterTPAssociation (cluster -> TrackingParticle) for the tracker validation.
# Phase-2 tracker: pixel + outer-tracker (Phase2TrackerCluster1D), no strips.
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import tpClusterProducer
process.tpClusterProducer = tpClusterProducer.clone(
    pixelClusterSrc=cms.InputTag("siPixelClusters"),
    phase2OTClusterSrc=cms.InputTag("siPhase2Clusters"),
    pixelSimLinkSrc=cms.InputTag("simSiPixelDigis", "Pixel"),
    phase2OTSimLinkSrc=cms.InputTag("simSiPixelDigis", "Tracker"),
    trackingParticleSrc=cms.InputTag("mix", "MergedTrackTruth"),
    throwOnMissingCollections=cms.bool(False),
)

process.trackerValidator = cms.EDAnalyzer(
    "BranchTrackerReplacementValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    tracks=cms.InputTag("generalTracks"),
    clusterTPMap=cms.InputTag("tpClusterProducer"),
)

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.default = cms.untracked.PSet(limit=cms.untracked.int32(0))
process.MessageLogger.cerr.BranchTruthReplacementValidator = cms.untracked.PSet(limit=cms.untracked.int32(-1))
process.MessageLogger.cerr.BranchTrackerReplacementValidator = cms.untracked.PSet(limit=cms.untracked.int32(-1))

process.p = cms.Path(
    process.truthGraphProducer
    + process.truthLogicalGraphProducer
    + process.detIdToRecHitMapProducer
    + process.truthLogicalGraphHitIndexProducer
    + process.tpClusterProducer
    + process.branchValidator
    + process.trackerValidator
)
