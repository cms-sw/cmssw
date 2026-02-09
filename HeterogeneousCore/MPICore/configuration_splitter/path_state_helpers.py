from collections import defaultdict, deque
from typing import Dict, Set, List

import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import *


# --- determine to which paths and sequences are the modules 

def _sequence_contains_module(seq: cms.Sequence, module_name: str) -> bool:
    """Check whether a cms.Sequence contains a module (recursively)."""
    try:
        return module_name in seq.moduleNames()
    except Exception:
        return False


def _path_contains_module(path: cms.Path, module_name: str) -> bool:
    """Check whether a cms.Path contains a module (directly or via sequences)."""
    try:
        return module_name in path.moduleNames()
    except Exception:
        return False


# def get_paths_and_sequences_of_the_modules(
#     process, modules
# ) -> Dict[str, Dict[str, List]]:
#     """
#     Determine which paths and sequences contain each module.

#     Returns:
#         {
#           module_name: {
#               "paths": [cms.Path, ...],
#               "sequences": [cms.Sequence, ...],
#           }
#         }
#     """

#     result: Dict[str, Dict[str, List]] = {}

#     sequences = getattr(process, "sequences", {})
#     paths = getattr(process, "paths", {})

#     for mod in modules:
#         result[mod] = {
#             "paths": [],
#             "sequences": [],
#         }

#         # --- sequences
#         for seq_name, seq in sequences.items():
#             if _sequence_contains_module(seq, mod):
#                 result[mod]["sequences"].append(seq_name)

#         # --- paths
#         for path_name, path in paths.items():
#             if _path_contains_module(path, mod):
#                 result[mod]["paths"].append(path_name)

#     return result


def get_sequences_of_the_modules(
    process, modules
) -> Dict[str, Dict[str, List[str]]]:
    """
    Determine which sequences directly depend on each module.
    Paths are intentionally ignored.

    Returns:
        {
          module_name: {
              "sequences": [sequence_name, ...],
          }
        }
    """

    result: Dict[str, Dict[str, List[str]]] = {}

    sequences = getattr(process, "sequences", {})

    for mod in modules:
        result[mod] = {"sequences": []}

        for seq_name, seq in sequences.items():
            try:
                deps = seq.directDependencies()
            except Exception:
                continue

            for kind, name in deps:
                if kind == "modules" and name == mod:
                    result[mod]["sequences"].append(seq_name)
                    break  # no need to scan further

    return result



def group_sequences(
    groups: List[List[str]],
    base_sequences: Dict[str, Dict[str, List[str]]],
) -> List[List[str]]:
    """
    For each dependency group, return the list of base sequences
    that directly contain any module in that group.

    Args:
        groups:
            List of dependency groups (each is a list of module names,
            ordered root -> leaf).
        base_sequences:
            Output of get_sequences_of_the_modules():
            {
              module_name: {
                  "sequences": [sequence_name, ...]
              }
            }

    Returns:
        List of lists of sequence names, one list per group.
    """

    sequences_by_group: List[List[str]] = []

    for group in groups:
        seqs = []
        seen = set()

        for module in group:
            info = base_sequences.get(module)
            if not info:
                continue

            for seq in info.get("sequences", []):
                if seq not in seen:
                    seen.add(seq)
                    seqs.append(seq)

        sequences_by_group.append(seqs)

    return sequences_by_group



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
