#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""HLT acceptance test: pyTICL's HLT target vs HLTIterTICLSequence.

The HLT_75e33 menu is a frozen confdb-style dump that can lag the live
producers.  pyTICL clones the *live* ``_cfi`` defaults, so it generates an
up-to-date, internally consistent HLT TICL config.  This test therefore asserts
the things pyTICL is responsible for and that must be exact:

  * the module SET (structure) matches HLTIterTICLSequence;
  * every PLUMBING parameter (InputTag / VInputTag wiring) matches.

and it *reports* the residual parameter-value deltas as "frozen-menu drift"
(parameters the live producers gained after the menu was frozen) -- a useful
signal that the menu should be regenerated, not a failure of pyTICL.
"""

import sys

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration import hlt_presets

# layer_clusters_barrel_tiles is set inconsistently in the frozen menu
# (CLUE3DHigh has the override, Recovery does not) and is unused without barrel.
PLUMBING_ALLOWLIST = {"layer_clusters_barrel_tiles"}


def _dump(value):
    return value.dumpPython() if value is not None else "<absent>"


def main():
    bp = cms.Process("TEST")
    bp.load("HLTrigger.Configuration.HLT_75e33.sequences.HLTIterTICLSequence_cfi")
    btask = bp.HLTIterTICLSequence

    tp = cms.Process("TEST")
    hlt_presets.v5_hlt().assemble().add_to_process(tp)
    ttask = tp.HLTIterTICLSequence   # the assembled top group (per the HLT target)

    base_labels = set(btask.moduleNames())
    test_labels = set(ttask.moduleNames())
    if base_labels != test_labels:
        print("STRUCTURE mismatch:")
        print("  missing :", sorted(base_labels - test_labels))
        print("  extra   :", sorted(test_labels - base_labels))
        return 1

    plumbing_problems = []
    drift = []
    for label in sorted(base_labels):
        bpars = getattr(bp, label).parameters_()
        tpars = getattr(tp, label).parameters_()
        for pname in sorted(set(bpars) | set(tpars)):
            bval, tval = bpars.get(pname), tpars.get(pname)
            if _dump(bval) == _dump(tval):
                continue
            is_plumbing = isinstance(bval, (cms.InputTag, cms.VInputTag)) or \
                isinstance(tval, (cms.InputTag, cms.VInputTag))
            if is_plumbing and pname not in PLUMBING_ALLOWLIST:
                plumbing_problems.append("%s.%s: menu=%s pyTICL=%s"
                                         % (label, pname, _dump(bval), _dump(tval)))
            else:
                drift.append("%s.%s" % (label, pname))

    print("Structure: OK (%d HLT modules match HLTIterTICLSequence)" % len(base_labels))
    if drift:
        print("\nFrozen-menu drift -- %d param(s) where the live producers differ from the "
              "frozen HLT_75e33 dump (pyTICL tracks the live cfi):" % len(drift))
        for d in drift:
            print("  - " + d)

    if plumbing_problems:
        print("\nPLUMBING mismatch (%d) -- these MUST match:" % len(plumbing_problems))
        for p in plumbing_problems:
            print("  - " + p)
        return 1

    print("\nOK: HLT structure + plumbing reproduced; residual deltas are frozen-menu drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
