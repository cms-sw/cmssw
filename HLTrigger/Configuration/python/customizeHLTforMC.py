import FWCore.ParameterSet.Config as cms

def customizeHLT_removeCPUOnlySequences_AlCaPFJet40(process):
    """
    Remove AlCa paths from the process to avoid running them twice 
    (Alpaka and SerialSync, CMSHLT-2461).
    """
    path_cpu_only = None
    path_standard = None

    # Identify target paths (handles potential version suffixes)
    for path_name in process.paths_():
        if path_name.startswith('AlCa_PFJet40_CPUOnly_v'):
            path_cpu_only = path_name
        elif path_name.startswith('AlCa_PFJet40_v'):
            path_standard = path_name

    # If CPUOnly path is not present, nothing to modify
    if not path_cpu_only:
        return process

    # If standard path is missing, copy fails
    if not path_standard:
        print("WARNING: path AlCa_PFJet40_CPUOnly found but not path AlCa_PFJet40, cannot remove it from the process")
        return process

    # Verify both prescalers exist before replacing
    has_pre_std = hasattr(process, 'hltPreAlCaPFJet40')
    has_pre_cpu = hasattr(process, 'hltPreAlCaPFJet40CPUOnly')

    if has_pre_std and has_pre_cpu:
        new_path = getattr(process, path_standard).copy()
        new_path.replace(process.hltPreAlCaPFJet40, process.hltPreAlCaPFJet40CPUOnly)
        setattr(process, path_cpu_only, new_path)
    else:
        print("WARNING: path AlCa_PFJet40_CPUOnly found but not hltPreAlCaPFJet40 or hltPreAlCaPFJet40CPUOnly, cannot replace it in the path")

    return process

def customizeHLTforMC(process):
  """adapt the HLT to run on MC, instead of data
  see Configuration/StandardSequences/Reconstruction_Data_cff.py
  which does the opposite, for RECO"""
  process = customizeHLT_removeCPUOnlySequences_AlCaPFJet40(process)
  return process
