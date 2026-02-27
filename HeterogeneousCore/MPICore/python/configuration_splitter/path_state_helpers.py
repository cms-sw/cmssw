from collections import defaultdict, deque
from typing import Dict, Set, List

import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import *


# functions to add activity filters and path state captures

def add_activity_filter(process, module_name):
    filter_object =  cms.EDFilter("PathStateRelease",
            state = cms.InputTag(module_name)
        )
    filter_name = f"activityFilter{module_name}"
    setattr(process, filter_name, filter_object)
    return filter_name



def insert_path_state_capture_before(
    process,
    first_modules_in_a_group,
    capture_name,
    prefix="PathStateCapture",
):
    """
      - create one PathStateCapture EDProducer
      - insert it at position 0 of each passed in sequence
    """

    # --- unique module name per group
    capture_name = f"{prefix}{capture_name}"

    # create the EDProducer
    setattr(
        process,
        capture_name,
        cms.EDProducer("PathStateCapture"),
    )

    capture = getattr(process, capture_name)

    # --- insert into sequences
    for module_name in first_modules_in_a_group:
        if not hasattr(process, module_name):
            print(f"[WARN] process has no sequence '{module_name}'")
            continue

        module = getattr(process, module_name)

        # insert at the beginning
        insert_modules_before(process, module, capture)
    
    return capture_name
