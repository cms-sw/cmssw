# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Standalone single-file driver for the Branch tracking DQM validator. The modules
# are the canonical ones from PhysicsTools.TruthInfo.truthGraphValidation_cff (the
# same sequence wired into globalValidation behind enableTruth) - this cfg only
# rebuilds the truth-graph chain from a GEN-SIM-DIGI-RECO file, runs the cluster->TP
# association and the BranchTrackingValidator, and writes a DQM root file (inspect
# under DQMData/Run 1/Tracking/Run summary/BranchValidator/TrackingParticle).

import FWCore.ParameterSet.Config as cms
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

# Canonical truth-graph + Branch-validator modules (single source of truth).
process.load("PhysicsTools.TruthInfo.truthGraphValidation_cff")

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(args.maxevts))
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(args.inputFile))
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False))

process.dqmOut = cms.OutputModule(
    "DQMRootOutputModule",
    fileName=cms.untracked.string(args.out),
)

process.p = cms.Path(
    process.truthGraphProducer
    + process.truthLogicalGraphProducer
    + process.simHitToRecHitMapProducer
    + process.truthLogicalGraphHitIndexProducer
    + process.truthTpClusterProducer
    + process.truthBranchTrackingAssociationProducer
    + process.branchTrackingValidator
)
process.e = cms.EndPath(process.dqmOut)
