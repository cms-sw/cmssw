#!/usr/bin/env python3
"""
Compare the P2GT conditions of two dumped menus (dumpL1TGTMenu.py output, or any
dumped cmsRun config) module by module. Structured menus are flattened first
(l1tGTMenuTools.flattenGTMenu), so a structured menu can be checked against a
flat reference, e.g. the dump of the menu before a refactoring:

    compareGTMenuDumps.py reference_fullConfigDump.py new_fullConfigDump.py

Exits with 1 if any condition or the algorithm list differs.
"""
import argparse
import difflib
import sys

import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTMenuTools import flattenGTMenu, is_gt_condition


def load_process(path):
    namespace = {}
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), namespace)
    return flattenGTMenu(namespace["process"])


def canonical(param):
    """Parameter value as nested python data, so that e.g. vdouble(25) and vdouble(25.0) compare equal"""
    if hasattr(param, "parameters_"):
        items = sorted((name, canonical(value)) for name, value in param.parameters_().items())
        return (type(param).__name__, getattr(param, "type_", lambda: "")(), tuple(items))
    if isinstance(param, cms.VPSet):
        return ("VPSet", tuple(canonical(p) for p in param))
    value = param.value()
    if isinstance(value, (list, tuple)):
        value = tuple(value)
    return (type(param).__name__, param.isTracked(), value)


def conditions(process):
    return {label: mod for label, mod in process.filters_().items() if is_gt_condition(mod)}


def algorithms(process):
    return sorted(canonical(pset) for pset in process.l1tGTAlgoBlockProducer.algorithms)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()

    ref, new = load_process(args.reference), load_process(args.candidate)
    ref_conds, new_conds = conditions(ref), conditions(new)

    problems = []
    for label in sorted(set(ref_conds) | set(new_conds)):
        if label not in new_conds:
            problems.append("missing in candidate: %s" % label)
        elif label not in ref_conds:
            problems.append("missing in reference: %s" % label)
        elif canonical(ref_conds[label]) != canonical(new_conds[label]):
            diff = difflib.unified_diff(ref_conds[label].dumpPython().splitlines(),
                                        new_conds[label].dumpPython().splitlines(),
                                        "reference", "candidate", lineterm="", n=2)
            problems.append("differs: %s\n%s" % (label, "\n".join(diff)))
    if algorithms(ref) != algorithms(new):
        problems.append("algorithm definitions differ")

    print("compared %d conditions" % len(ref_conds))
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
