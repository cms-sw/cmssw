import FWCore.ParameterSet.Config as cms

import argparse

parser = argparse.ArgumentParser(description="Test storage proxies")
parser.add_argument("--trace", action="store_true", help="Enable StorageTraceProxy")
parser.add_argument("--latencyRead", action="store_true", help="Add read latency")
parser.add_argument("--latencyWrite", action="store_true", help="Add write latency")
args = parser.parse_args()

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:test.root")
)

process.out = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("output.root"))

adaptor = cms.Service("AdaptorConfig", storageProxies = cms.untracked.VPSet())
if args.latencyRead:
    adaptor.storageProxies.append(cms.PSet(
        type = cms.untracked.string("StorageAddLatencyProxy"),
        read = cms.untracked.uint32(100),
        readv = cms.untracked.uint32(100),
    ))
if args.latencyWrite:
    adaptor.storageProxies.append(cms.PSet(
        type = cms.untracked.string("StorageAddLatencyProxy"),
        write = cms.untracked.uint32(100),
        writev = cms.untracked.uint32(100),
    ))
if args.trace:
    adaptor.storageProxies.append(cms.PSet(
        type = cms.untracked.string("StorageTracerProxy"),
        traceFilePattern = cms.untracked.string("trace_%I.txt"),
    ))

process.add_(adaptor)

process.ep = cms.EndPath(process.out)
