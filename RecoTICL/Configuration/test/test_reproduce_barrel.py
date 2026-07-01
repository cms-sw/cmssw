#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""M3: pyTICL reproduces the TICL barrel iteration byte-for-byte.

Compares the barrel modules pyTICL generates (filteredLayerClustersCLUE3DBarrel,
ticlTrackstersCLUE3DBarrel, ticlLayerTileBarrel) against the baseline defined in
RecoHGCal/TICL (CLUE3DBarrel_cff + iterativeTICL_cff)."""

import sys

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration import presets

BARREL_MODULES = [
    "filteredLayerClustersCLUE3DBarrel",
    "ticlTrackstersCLUE3DBarrel",
    "ticlLayerTileBarrel",
]


def main():
    base = cms.Process("TEST")
    base.load("RecoHGCal.TICL.iterativeTICL_cff")

    gen = cms.Process("TEST")
    presets.barrel().assemble().add_to_process(gen)

    ok = True
    for label in BARREL_MODULES:
        b = getattr(base, label).dumpPython()
        g = getattr(gen, label).dumpPython()
        if b != g:
            print("MODULE DIFFERS: %s" % label)
            import difflib
            for line in difflib.unified_diff(b.splitlines(), g.splitlines(),
                                             fromfile="baseline", tofile="pyTICL", lineterm=""):
                print("  " + line)
            ok = False
        else:
            print("OK   : %s reproduced byte-for-byte" % label)

    if not ok:
        return 1
    print("\nOK: pyTICL reproduces the TICL barrel iteration byte-for-byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
