#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""M2: validate the single-source-of-truth derivation of labels, associators,
dumper and validator against the live baselines, byte-for-byte where possible.

Note: ``RecoHGCal.TICL.ticlDumper_cff`` currently NameErrors on import (it
references an undefined ``ticlIterLabels`` inside a barrel ``toModify``), so the
dumper is checked structurally against the documented derivation instead.
"""

import sys

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration import presets, validation


def main():
    from RecoHGCal.TICL.iterativeTICL_cff import associatorsInstances as base_inst, ticlIterLabelsPSet
    from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi \
        import allTrackstersToSimTrackstersAssociationsByLCs as base_bylcs
    from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociationByHits_cfi \
        import allTrackstersToSimTrackstersAssociationsByHits as base_byhits
    from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidator as base_validator

    labels = list(ticlIterLabelsPSet.labels)
    ok = True

    # pyTICL derives the v5 labels from the config alone
    derived = validation.default_validation_labels(presets.v5())
    if derived != labels:
        print("FAIL: derived labels %s != baseline %s" % (derived, labels)); ok = False
    else:
        print("OK   : derived validation labels == ticlIterLabelsPSet.labels")

    # associator instance names
    if validation.associator_instances(labels) != list(base_inst):
        print("FAIL: associatorsInstances mismatch"); ok = False
    else:
        print("OK   : associatorsInstances (%d) match baseline" % len(base_inst))

    # associator producers, byte-for-byte
    for name, built, base in [
        ("ByLCs", validation.build_associators_by_lcs(labels), base_bylcs),
        ("ByHits", validation.build_associators_by_hits(labels), base_byhits),
        ("hgcalValidator", validation.build_hgcal_validator(labels), base_validator),
    ]:
        if built.dumpPython() != base.dumpPython():
            print("FAIL: %s does not match baseline byte-for-byte" % name)
            import difflib
            for line in list(difflib.unified_diff(base.dumpPython().splitlines(),
                                                  built.dumpPython().splitlines(),
                                                  lineterm=""))[:25]:
                print("    " + line)
            ok = False
        else:
            print("OK   : %s reproduced byte-for-byte" % name)

    # dumper: structural (cff baseline is broken on import)
    dumper = validation.build_ticl_dumper(labels)
    n_tc = len(dumper.tracksterCollections)
    n_assoc = len(dumper.associators)
    if n_tc != len(labels) + 2 or n_assoc != len(labels) * 2:
        print("FAIL: dumper structure: tracksterCollections=%d (want %d), associators=%d (want %d)"
              % (n_tc, len(labels) + 2, n_assoc, len(labels) * 2)); ok = False
    else:
        print("OK   : ticlDumper structure (%d collections, %d associators)" % (n_tc, n_assoc))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
