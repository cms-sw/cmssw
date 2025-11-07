#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Exporter round-trip test.

Export the v5 config to a cff fragment, load it back, and assert the reloaded
``iterTICLTask`` reproduces the baseline byte-for-byte.  This proves the emitted
cff is self-contained and faithful (the property HLT relies on).
"""

import importlib.util
import os
import sys
import tempfile

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration import presets
from RecoTICL.Configuration.compare import diff_tasks


def _load_fragment(path):
    spec = importlib.util.spec_from_file_location("pyticl_exported_cff", path)
    frag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frag)
    process = cms.Process("TEST")
    for name in dir(frag):
        obj = getattr(frag, name)
        if isinstance(obj, cms._Module):
            setattr(process, name, obj)
    for name in dir(frag):
        obj = getattr(frag, name)
        if isinstance(obj, cms.Task):
            setattr(process, name, obj)
    return process


def main():
    tmp = tempfile.mkdtemp(prefix="pyticl_")
    path = os.path.join(tmp, "exported_cff.py")
    presets.v5().to_cff(path)

    gen = _load_fragment(path)
    base = cms.Process("TEST")
    base.load("RecoHGCal.TICL.iterativeTICL_cff")

    diff = diff_tasks(base, base.iterTICLTask, gen, gen.iterTICLTask)
    if diff:
        print("Exported cff does NOT round-trip to the baseline:\n")
        print(diff)
        return 1
    print("OK: exported cff round-trips to iterTICLTask byte-for-byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
