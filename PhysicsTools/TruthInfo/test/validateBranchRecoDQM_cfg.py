# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Standalone single-file driver for the generic reco-side Branch validators. The
# modules are the canonical ones from PhysicsTools.TruthInfo.truthGraphValidation_cff
# (truthGraphRecoSideValidationSequence) - this cfg rebuilds the truth-graph chain
# from a GEN-SIM-RECO file and runs BranchTrackRecoValidator (reco tracks) and
# BranchTracksterRecoValidator (TICL tracksters), writing a DQMIO file (inspect under
# DQMData/Run 1/{Tracking,HGCAL}/Run summary/BranchValidator).
#
# EXPERIMENTAL / opt-in: this is why these two validators are NOT in the default
# globalValidation sequence. The reco-side efficiency/fake/merge/duplicate is only
# meaningful against a DISJOINT (antichain) interesting-particle reference - a Branch
# subgraph aggregates descendants, so against the full graph every reco object merges
# nested branches (merge-rate ~1, efficiency ~0). A flat PDG-id list is a sufficient
# antichain only for non-showering species: the track validator is restricted to
# muons (clean on e.g. Z->mumu); the trackster validator's HGCAL-entering PDG-id list
# still has residual nesting and is degenerate for showering particles. The proper
# reference is the BranchSelector antichain (not yet wired).

import FWCore.ParameterSet.Config as cms
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

# Canonical truth-graph + reco-side Branch-validator modules (single source of truth).
process.load("PhysicsTools.TruthInfo.truthGraphValidation_cff")

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(args.maxevts))
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(args.inputFile))
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False))

process.dqmOut = cms.OutputModule("DQMRootOutputModule", fileName=cms.untracked.string(args.out))

process.p = cms.Path(
    process.truthGraphProducer
    + process.truthLogicalGraphProducer
    + process.detIdToRecHitMapProducer
    + process.truthLogicalGraphHitIndexProducer
    + process.truthGraphRecoSideValidationSequence
)
process.e = cms.EndPath(process.dqmOut)
