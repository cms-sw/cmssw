#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""M2: validate the single-source-of-truth derivation of labels, associators,
dumper and validator against the live baselines, byte-for-byte.
"""

import importlib.util
import sys

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration import presets, validation
from RecoTICL.Configuration.target import HLT


def main():
    from RecoTICL.Configuration.iterativeTICL_cff import associatorsInstances as base_inst, ticlIterLabelsPSet
    from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi \
        import allTrackstersToSimTrackstersAssociationsByLCs as base_bylcs
    from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociationByHits_cfi \
        import allTrackstersToSimTrackstersAssociationsByHits as base_byhits
    from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidator as base_validator
    from Validation.TICLValidation.ticlDumper_cff import ticlDumper as base_dumper

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
        ("ticlDumper", validation.build_ticl_dumper(labels), base_dumper),
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

    # HLT target: the supercluster label names a module of the HLT menu
    sc_label = HLT.supercluster_dnn_label
    if importlib.util.find_spec("HLTrigger.Configuration.HLT_75e33.modules.%s_cfi" % sc_label) is None:
        print("FAIL: HLT supercluster label %s is not a module of HLT_75e33" % sc_label); ok = False
    else:
        print("OK   : HLT supercluster label %s is a module of HLT_75e33" % sc_label)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
