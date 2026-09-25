import FWCore.ParameterSet.Config as cms

# This fragment is intended to collect all ProcessAccelerator objects
# used in production

from HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi import ProcessAcceleratorCUDA
from HeterogeneousCore.ROCmCore.ProcessAcceleratorROCm_cfi import ProcessAcceleratorROCm
from HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi import ProcessAcceleratorAlpaka

# Update the environment variables used by MPI to select the accelerator
# backend
from HeterogeneousCore.MPIServices.ProcessAcceleratorMPI_cfi import ProcessAcceleratorMPI
