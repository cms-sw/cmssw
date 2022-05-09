import FWCore.ParameterSet.Config as cms

# customisation for running the Patatrack reconstruction, common parts
def customiseCommon(process):
    print('# WARNING: customizeHLTforPatatrack.customiseCommon is deprecated. No changes applied to the configuration.')
    return process

# customisation for running the "Patatrack" pixel local reconstruction
def customisePixelLocalReconstruction(process):
    print('# WARNING: customizeHLTforPatatrack.customisePixelLocalReconstruction is deprecated. No changes applied to the configuration.')
    return process

# customisation for running the "Patatrack" pixel track reconstruction
def customisePixelTrackReconstruction(process):
    print('# WARNING: customizeHLTforPatatrack.customisePixelTrackReconstruction is deprecated. No changes applied to the configuration.')
    return process

# customisation for offloading the ECAL local reconstruction via CUDA if a supported gpu is present
def customiseEcalLocalReconstruction(process):
    print('# WARNING: customizeHLTforPatatrack.customiseEcalLocalReconstruction is deprecated. No changes applied to the configuration.')
    return process

# customisation for offloading the HCAL local reconstruction via CUDA if a supported gpu is present
def customiseHcalLocalReconstruction(process):
    print('# WARNING: customizeHLTforPatatrack.customiseHcalLocalReconstruction is deprecated. No changes applied to the configuration.')
    return process

# customisation to enable pixel triplets instead of quadruplets
def enablePatatrackPixelTriplets(process):
    print('# WARNING: customizeHLTforPatatrack.enablePatatrackPixelTriplets is deprecated. No changes applied to the configuration.')
    return process

# customisation for running the Patatrack reconstruction, with automatic offload via CUDA when a supported gpu is available
def customizeHLTforPatatrack(process):
    print('# WARNING: customizeHLTforPatatrack.customizeHLTforPatatrack is deprecated. No changes applied to the configuration.')
    return process

# customisation for running the Patatrack triplets reconstruction, with automatic offload via CUDA when a supported gpu is available
def customizeHLTforPatatrackTriplets(process):
    print('# WARNING: customizeHLTforPatatrack.customizeHLTforPatatrackTriplets is deprecated. No changes applied to the configuration.')
    return process
