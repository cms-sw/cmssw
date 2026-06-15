# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Standalone driver for the generic reco-side Branch validators: rebuilds the
# truth-graph chain from a GEN-SIM-RECO file and runs BranchTrackRecoValidator
# (reco tracks) and BranchTracksterRecoValidator (TICL tracksters), writing a DQMIO
# file (inspect under DQMData/Run 1/{Tracking,HGCAL}/Run summary/BranchValidator).
#
# EXPERIMENTAL: the reco-side efficiency/fake/merge/duplicate is only meaningful
# against a DISJOINT (antichain) interesting-particle reference - a Branch subgraph
# aggregates descendants, so against the full graph every reco object merges nested
# branches (merge-rate ~1, efficiency ~0). A flat PDG-id list is a sufficient
# antichain only for non-showering species: the track validator below is restricted
# to muons (clean on e.g. Z->mumu); the trackster validator's HGCAL-entering PDG-id
# list still has residual nesting and is degenerate for showering particles. The
# proper reference is the BranchSelector antichain (not yet wired).

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("inputFile", nargs='?', default="step3.root", metavar='FILE')
parser.add_argument('-n', "--maxevts", type=int, default=5)
parser.add_argument('-o', "--out", default="branch_reco_dqm.root")
args = parser.parse_args()
if '/' not in args.inputFile and ':' not in args.inputFile:
    args.inputFile = 'file:' + args.inputFile

process = cms.Process("BRANCHRECODQM")
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

process.branchTrackRecoValidator = DQMEDAnalyzer(
    "BranchTrackRecoValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    recoCollection=cms.InputTag("generalTracks"),
    # Muons only: a near-antichain (muons do not shower) so the reco-side metrics are
    # meaningful. Broader charged-stable lists re-introduce the nesting degeneracy.
    interestingPdgIds=cms.vint32(13, -13),
    folder=cms.string("Tracking/BranchValidator/recoTrack"),
    xName=cms.string("pt"),
    xTitle=cms.string("p_{T} [GeV]"),
    xMax=cms.double(200.0),
    minX=cms.double(0.9),
    minAbsEta=cms.double(0.0),
    maxAbsEta=cms.double(3.0),
    matchThreshold=cms.double(0.5),
    mergeThreshold=cms.double(0.3),
)

process.branchTracksterRecoValidator = DQMEDAnalyzer(
    "BranchTracksterRecoValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    recoCollection=cms.InputTag("ticlTrackstersCLUE3DHigh"),
    layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
    # Disjoint reference set (see truthGraphValidation_cff): HGCAL-entering species.
    interestingPdgIds=cms.vint32(22, 11, -11, 211, -211, 321, -321, 2212, -2212, 2112),
    folder=cms.string("HGCAL/BranchValidator/Trackster"),
    xName=cms.string("energy"),
    xTitle=cms.string("E [GeV]"),
    xMax=cms.double(500.0),
    minX=cms.double(1.0),
    minAbsEta=cms.double(1.5),
    maxAbsEta=cms.double(3.0),
    matchThreshold=cms.double(0.5),
    mergeThreshold=cms.double(0.3),
)

process.dqmOut = cms.OutputModule("DQMRootOutputModule", fileName=cms.untracked.string(args.out))

process.p = cms.Path(
    process.truthGraphProducer
    + process.truthLogicalGraphProducer
    + process.simHitToRecHitMapProducer
    + process.truthLogicalGraphHitIndexProducer
    + process.branchTrackRecoValidator
    + process.branchTracksterRecoValidator
)
process.e = cms.EndPath(process.dqmOut)
