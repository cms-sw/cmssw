#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""DOT graph export test.

Build the v5 config, emit its module graph, and assert the graph has the
expected structure: the core TICL modules as nodes, at least one external
upstream input, and typed edges that carry the consumed/produced product types.
"""

import os
import sys
import tempfile

from RecoTICL.Configuration import presets
from RecoTICL.Configuration.graph import build_graph, to_dot


def main():
    cfg = presets.v5()
    assembled = cfg.assemble()
    nodes, edges = build_graph(assembled)

    # the core modules must appear as nodes
    for label in ("ticlTrackstersCLUE3DHigh", "ticlTracksterLinks", "ticlCandidate"):
        if label not in nodes:
            print("missing expected node: %s" % label)
            return 1

    # at least one external upstream input (e.g. hgcalMergeLayerClusters) must appear
    if not any(n["kind"] == "external" for n in nodes.values()):
        print("expected at least one external input node")
        return 1

    # every edge must carry a C++ product type and target a known module
    if not edges:
        print("no edges built")
        return 1
    for e in edges:
        if not e["cpp_type"] or e["dst"] not in nodes:
            print("bad edge: %r" % e)
            return 1

    # a known typed connection: the links module consumes vector<vector<Trackster>>
    links_in = [e for e in edges if e["dst"] == "ticlTracksterLinks" and "Trackster" in e["cpp_type"]]
    if not links_in:
        print("expected a Trackster edge into ticlTracksterLinks")
        return 1

    dot = to_dot(cfg)
    for needle in ("digraph pyTICL", "ticlTrackstersCLUE3DHigh", "->"):
        if needle not in dot:
            print("DOT missing %r" % needle)
            return 1

    # writing to a file works and is non-empty
    path = os.path.join(tempfile.mkdtemp(prefix="pyticl_"), "graph.dot")
    to_dot(cfg, path)
    if not os.path.getsize(path):
        print("empty DOT file")
        return 1

    print("OK: %d nodes, %d edges; DOT written to %s" % (len(nodes), len(edges), path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
