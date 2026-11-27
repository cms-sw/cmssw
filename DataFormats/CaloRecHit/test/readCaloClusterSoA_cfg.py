# Reads the reco::CaloClusterHostCollection of writeCaloClusterSoA_cfg.py and checks its content.
import argparse
import sys

import FWCore.ParameterSet.Config as cms

parser = argparse.ArgumentParser(prog=sys.argv[0], description="Read and check a reco::CaloClusterHostCollection")
parser.add_argument("--rntuple", action="store_true", help="read the RNTuple file instead of the TTree file")
args = parser.parse_args()

process = cms.Process("READ")
if args.rntuple:
    process.source = cms.Source("RNTupleTempSource", fileNames = cms.untracked.vstring("file:testCaloClusterSoA_rntuple.root"))
else:
    process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:testCaloClusterSoA.root"))

process.reader = cms.EDAnalyzer("TestReadCaloClusterSoA", source = cms.InputTag("writer"))

process.path = cms.Path(process.reader)
