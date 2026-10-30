#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
"""The GEN-only graph of a two-particle gun holds the two GEN particles and no SIM node, and
the run metadata describes a GEN-only gun job."""

import os
import sys

import ROOT
from DataFormats.FWLite import Events, Handle, Runs
from PhysicsTools.TruthInfo.graphTools import LEVEL_BITS

graph = Handle("truth::Graph")
nEvents = 0
for event in Events(sys.argv[1]):
    event.getByLabel("truthGenOnlyLogicalGraphProducer", graph)
    particles = graph.product().particles()
    pdgIds = sorted(particles[i].pdgId for i in range(particles.size()))
    if pdgIds != [-211, 211]:
        sys.exit("event %d: GEN particles %s, expected [-211, 211]" % (nEvents, pdgIds))
    if any(particles[i].simNode >= 0 or particles[i].genNode < 0 for i in range(particles.size())):
        sys.exit("event %d: the GEN-only graph holds a SIM particle" % nEvents)
    nEvents += 1
if nEvents != 5:
    sys.exit("read %d events, expected 5" % nEvents)

# One metadata product for the one run, which describes the gun job.
metadata = Handle("truth::GraphMetadata")
products = []
for run in Runs(sys.argv[1]):
    run.getByLabel("truthGraphMetadata", metadata)
    products.append(metadata.product())
if len(products) != 1:
    sys.exit("read %d metadata products, expected 1" % len(products))
meta = products[0]
expected = {
    "buildMode": int(ROOT.truth.GraphMetadata.BuildMode.GenOnly),
    "generatorModule": "FlatRandomPtGunProducer",
    "cmsswVersion": os.environ["CMSSW_VERSION"],
    "levelNames": len(LEVEL_BITS),
    "hitIndexChannels": 0,
    "dropHitlessSimSubgraphs": False,
}
found = {
    "buildMode": int(meta.buildMode),
    "generatorModule": str(meta.generatorModule),
    "cmsswVersion": str(meta.cmsswVersion),
    "levelNames": meta.levelNames.size(),
    "hitIndexChannels": meta.hitIndexChannels.size(),
    "dropHitlessSimSubgraphs": bool(meta.dropHitlessSimSubgraphs),
}
if found != expected:
    sys.exit("metadata %s, expected %s" % (found, expected))
levels = {str(meta.levelNames[i]): meta.levelFlags[i] for i in range(meta.levelNames.size())}
if levels != LEVEL_BITS:
    sys.exit("metadata level bits %s, expected %s" % (levels, LEVEL_BITS))
print("OK: %d events, two GEN particles each, no SIM node, metadata %s" % (nEvents, found))
