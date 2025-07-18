"""
Module for handling L1 trigger menu constants and conversions.
"""

import FWCore.ParameterSet.Config as cms
from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024_explicitSeeds.l1tGTObject_scalings import scalings
from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024_explicitSeeds.l1tGTObject_ids import objectIDs

obj_regions_abseta_lowbounds = {
    "CL2Photons": { "barrel": 0, "endcap": 1.479 },
    "CL2Electrons": { "barrel": 0, "endcap": 1.479 },

    "CL2Taus": { "barrel": 0, "endcap": 1.5 },
    "CL2JetsSC4": { "barrel": 0, "endcap": 1.5, "forward": 2.4 },

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
