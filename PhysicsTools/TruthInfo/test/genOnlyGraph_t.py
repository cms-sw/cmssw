#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
"""The GEN-only graph of a two-particle gun holds the two GEN particles and no SIM node."""

import sys

from DataFormats.FWLite import Events, Handle

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
print("OK: %d events, two GEN particles each, no SIM node" % nEvents)
