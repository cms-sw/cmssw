#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Drift detector: lock the catalog against the live ``_cfi`` defaults.

For every module in the catalog this loads the *real* default and compares the
set of ``InputTag`` / ``VInputTag`` parameters (the plumbing) against what the
catalog declares (``consumes`` + ``external_inputs``).  If a producer gains a
new InputTag parameter, drops one, or changes a single<->vector kind, this test
fails -- telling the maintainer that the raw configuration changed and the
catalog (and connection rules) must be updated.
"""

import sys

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.catalog import CATALOG


def _import_default(spec):
    mod = __import__(spec.cfi_module, fromlist=[spec.cfi_symbol])
    return getattr(mod, spec.cfi_symbol)


def _input_params(module):
    """{param: is_vector} for every top-level InputTag/VInputTag parameter."""
    found = {}
    for name, value in module.parameters_().items():
        if isinstance(value, cms.VInputTag):
            found[name] = True
        elif isinstance(value, cms.InputTag):
            found[name] = False
    return found


def main():
    problems = []
    for key, spec in sorted(CATALOG.items()):
        module = _import_default(spec)
        live = _input_params(module)
        declared_vec = {c.param: c.vector for c in spec.consumes}
        accounted = set(declared_vec) | set(spec.external_inputs)

        for param in sorted(set(live) - accounted):
            problems.append("%s: NEW unaccounted InputTag parameter %r -- update "
                            "catalog consumes/external_inputs + connection rules"
                            % (key, param))
        for param in sorted(accounted - set(live)):
            problems.append("%s: catalog declares InputTag %r but the live cfi no "
                            "longer has it (renamed/removed?)" % (key, param))
        for param in sorted(set(declared_vec) & set(live)):
            if declared_vec[param] != live[param]:
                problems.append("%s.%s: single/vector mismatch (catalog vector=%s, "
                                "live vector=%s)" % (key, param, declared_vec[param],
                                                     live[param]))

    if problems:
        print("Catalog/schema drift detected:\n  - " + "\n  - ".join(problems))
        return 1
    print("OK: catalog matches the live cfi defaults for %d modules" % len(CATALOG))
    return 0


if __name__ == "__main__":
    sys.exit(main())
