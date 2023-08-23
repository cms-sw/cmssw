import FWCore.ParameterSet.Config as cms

# This fragment is intended to collect all ProcessAccelerator objects
# used in production

from HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi import ProcessAcceleratorCUDA
from HeterogeneousCore.ROCmCore.ProcessAcceleratorROCm_cfi import ProcessAcceleratorROCm
