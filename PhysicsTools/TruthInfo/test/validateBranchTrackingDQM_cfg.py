# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Standalone driver for the Branch tracking DQM validator: rebuilds the truth-graph
# chain from a GEN-SIM-DIGI-RECO file, runs tpClusterProducer (cluster->TP) and the
# BranchTrackingValidator, and writes a DQM root file (inspect under
# DQMData/Run 1/Tracking/Run summary/BranchValidator/TrackingParticle).

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("inputFile", nargs='?', default="step3.root", metavar='FILE')
parser.add_argument('-n', "--maxevts", type=int, default=5)
parser.add_argument('-o', "--out", default="branch_tracking_dqm.root")
args = parser.parse_args()
if '/' not in args.inputFile and ':' not in args.inputFile:
    args.inputFile = 'file:' + args.inputFile

process = cms.Process("BRANCHTRKDQM")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryExtendedRun4D120Reco_cff")
process.load("DQMServices.Core.DQMStore_cfi")
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
        seedPdgIds=cms.vint32(),
        seedHadronFlavors=cms.vint32(),
        seedParentDepth=cms.uint32(0),
        keepStableSpectators=cms.bool(True),
        decayPdgIdGroups=cms.VPSet(),
        ignoredPdgIds=cms.vint32(),
        ignoredParticleIds=cms.vuint32(),
    ),
)

process.simHitToRecHitMapProducer = cms.EDProducer(
    "SimHitToRecHitMapProducer",
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

# trackerSimHitCollections left at the C++ default (filled from fillDescriptions).
process.truthLogicalGraphHitIndexProducer = cms.EDProducer(
    "TruthLogicalGraphHitIndexProducer",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    recHitMap=cms.InputTag("simHitToRecHitMapProducer"),
    simHitCollections=cms.VInputTag(
        cms.InputTag("g4SimHits", "HGCHitsEE"),
        cms.InputTag("g4SimHits", "HGCHitsHEfront"),
        cms.InputTag("g4SimHits", "HGCHitsHEback"),
        cms.InputTag("g4SimHits", "EcalHitsEB"),
        cms.InputTag("g4SimHits", "HcalHits"),
    ),
    doHGCalRelabelling=cms.bool(False),
)

# ClusterTPAssociation (cluster -> TrackingParticle); Phase-2 pixel + outer tracker.
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import tpClusterProducer
process.tpClusterProducer = tpClusterProducer.clone(
    pixelClusterSrc=cms.InputTag("siPixelClusters"),
    phase2OTClusterSrc=cms.InputTag("siPhase2Clusters"),
    pixelSimLinkSrc=cms.InputTag("simSiPixelDigis", "Pixel"),
    phase2OTSimLinkSrc=cms.InputTag("simSiPixelDigis", "Tracker"),
    trackingParticleSrc=cms.InputTag("mix", "MergedTrackTruth"),
    throwOnMissingCollections=cms.bool(False),
)

process.branchTrackingValidator = DQMEDAnalyzer(
    "BranchTrackingValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("truthGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    tracks=cms.InputTag("generalTracks"),
    clusterTPMap=cms.InputTag("tpClusterProducer"),
    folder=cms.string("Tracking/BranchValidator"),
    minPt=cms.double(0.9),
    maxEta=cms.double(3.0),
)

process.dqmOut = cms.OutputModule(
    "DQMRootOutputModule",
    fileName=cms.untracked.string(args.out),
)

process.p = cms.Path(
    process.truthGraphProducer
    + process.truthLogicalGraphProducer
    + process.simHitToRecHitMapProducer
    + process.truthLogicalGraphHitIndexProducer
    + process.tpClusterProducer
    + process.branchTrackingValidator
)
process.e = cms.EndPath(process.dqmOut)
