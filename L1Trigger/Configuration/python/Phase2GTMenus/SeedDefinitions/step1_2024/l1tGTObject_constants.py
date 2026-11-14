"""
Module for handling L1 trigger menu constants and conversions.
"""

import FWCore.ParameterSet.Config as cms
from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTObject_scalings import scalings
from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTObject_ids import objectIDs

obj_regions_abseta_lowbounds = {
    "CL2Photons": { "barrel": 0, "endcap": 1.479 },
    "CL2Electrons": { "barrel": 0, "endcap": 1.479 },
    "L1EG": { "barrel": 0, "endcap": 1.479 },

    "CL2Taus": { "barrel": 0, "endcap": 1.5 },
    "CL2JetsSC4": { "barrel": 0, "endcap": 1.5, "forwardHGC": 2.4, "forwardHF": 3.0 },
    "CL2JetsSC8": { "barrel": 0, "endcap": 1.5, "forwardHGC": 2.4, "forwardHF": 3.0 },

    "GMTTkMuons": { "barrel": 0, "overlap": 0.83, "endcap": 1.24 },
    "GMTMuons": { "barrel": 0, "overlap": 0.83, "endcap": 1.24 },

    "CL2HtSum": {"inclusive": 0},
    "CL2EtSum": {"inclusive": 0},
}

def get_object_etalowbounds(obj):
    return cms.vdouble(tuple(obj_regions_abseta_lowbounds[obj].values()))

def off2onl_thresholds(thr, obj, id, region, scalings=scalings):
    """
    Convert offline thresholds to online thresholds.

    Args:
        thr (float): The offline threshold.
        obj (str): The object type.
        id (str): The object ID.
        region (str): The region.
        scalings (dict): The scalings dictionary.

    Returns:
        float: The online threshold.
    """
    offset = scalings[obj][id][region]["offset"]
    slope = scalings[obj][id][region]["slope"]
    new_thr = round((thr - offset) / slope, 1)

    if "Jet" in obj:
        # Safety cut
        return max(25, new_thr)
    else:
        return max(0, new_thr)

def get_object_thrs(thr, obj, id = "default", scalings=scalings):
    regions = obj_regions_abseta_lowbounds[obj].keys()
    thresholds = [off2onl_thresholds(thr, obj, id, region) for region in regions]
    if len(thresholds) > 1:
        return cms.vdouble(tuple(thresholds))
    else:
        return cms.double(thresholds[0])

def get_object_ids(obj, id = "default", obj_dict=objectIDs):
    values = obj_dict[obj][id]["qual"]
    if isinstance(values, dict):
        regions = obj_regions_abseta_lowbounds[obj].keys()
        return cms.vuint32(tuple(values[region] for region in regions))
    else:
        return cms.uint32(values)

def get_object_isos(obj, id = "default", obj_dict=objectIDs):
    values = obj_dict[obj][id]["iso"]
    if isinstance(values, dict):
        regions = obj_regions_abseta_lowbounds[obj].keys()
        return cms.vdouble(tuple(values[region] for region in regions))
    else:
        return cms.double(values)

############################################################
# Structured (ConfDB-friendly) object/scaling definitions
############################################################

def gt_ref(name):
    """Reference to a top-level PSet (object definition or pT scaling), resolved by the framework.

    Used as `object = gt_ref("l1tGTtkMuonVLoose")` or `ptScaling = gt_ref("l1tGTScaling_GMTTkMuons_VLoose")`
    in a condition/collection, so that shared definitions stay single, editable PSets (e.g. in ConfDB).
    """
    return cms.PSet(refToPSet_ = cms.string(name))

def get_scaling_name(obj, id = "default"):
    return "l1tGTScaling_%s_%s" % (obj, id)

def get_object_scaling(obj, id = "default", scalings=scalings):
    """Offline pT scaling of an object ID as a PSet, ordered like regionsAbsEtaLowerBounds.

    The online threshold is computed by the emulator as
    max(minOnlinePt, round((offline - offset) / slope, 1)), see off2onl_thresholds.
    """
    regions = obj_regions_abseta_lowbounds[obj].keys()
    pset = cms.PSet(
        regionsOffset = cms.vdouble(tuple(scalings[obj][id][region]["offset"] for region in regions)),
        regionsSlope = cms.vdouble(tuple(scalings[obj][id][region]["slope"] for region in regions)),
    )
    if "Jet" in obj:
        pset.minOnlinePt = cms.double(25) # safety cut, as in off2onl_thresholds
    return pset
