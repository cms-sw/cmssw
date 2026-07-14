import FWCore.ParameterSet.Config as cms

def customizeHLT_removeCPUOnlySequences_AlCaPFJet40(process):
    """
    Replace the duplicated CPU-only (SerialSync) reconstruction in the
    AlCa_PFJet40_CPUOnly path with the standard AlCa_PFJet40 reconstruction
    to avoid running particle-flow reconstruction twice in MC (CMSHLT-2461).    
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
        print("WARNING: found AlCa_PFJet40_CPUOnly path but no AlCa_PFJet40 path; cannot replace CPUOnly reconstruction")
        return process

    # Verify both prescalers exist before replacing
    has_pre_std = hasattr(process, 'hltPreAlCaPFJet40')
    has_pre_cpu = hasattr(process, 'hltPreAlCaPFJet40CPUOnly')

    if has_pre_std and has_pre_cpu:
        new_path = getattr(process, path_standard).copy()
        new_path.replace(process.hltPreAlCaPFJet40, process.hltPreAlCaPFJet40CPUOnly)
        setattr(process, path_cpu_only, new_path)
    else:
        print("WARNING: found AlCa_PFJet40_CPUOnly path but missing hltPreAlCaPFJet40 and/or hltPreAlCaPFJet40CPUOnly; leaving CPUOnly path unchanged")

    return process

def customizeHLTforMC(process):
  """adapt the HLT to run on MC, instead of data
  see Configuration/StandardSequences/Reconstruction_Data_cff.py
  which does the opposite, for RECO"""
  process = customizeHLT_removeCPUOnlySequences_AlCaPFJet40(process)
  return process
