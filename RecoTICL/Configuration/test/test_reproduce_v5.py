#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""M1 acceptance gate + primary drift detector.

Assert that the configuration pyTICL generates for the v5 default reproduces the
live baseline ``iterTICLTask`` (from ``RecoHGCal.TICL.iterativeTICL_cff``)
byte-for-byte, module by module.  If anyone edits a baseline cff in a way pyTICL
does not mirror, this test fails with a precise diff.
"""

import sys

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration import presets
from RecoTICL.Configuration.compare import diff_tasks


def build_baseline():
    p = cms.Process("TEST")
    p.load("RecoHGCal.TICL.iterativeTICL_cff")
    return p, p.iterTICLTask


def build_pyticl():
    p = cms.Process("TEST")
    assembled = presets.v5().assemble()
    assembled.add_to_process(p)
    return p, p.iterTICLTask


def main():
    base_p, base_task = build_baseline()
    test_p, test_task = build_pyticl()
    diff = diff_tasks(base_p, base_task, test_p, test_task)
    if diff:
        print("pyTICL v5 config does NOT match baseline iterTICLTask:\n")
        print(diff)
        return 1
    n = len(base_task.moduleNames())
    print("OK: pyTICL v5 reproduces iterTICLTask byte-for-byte (%d modules)" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
