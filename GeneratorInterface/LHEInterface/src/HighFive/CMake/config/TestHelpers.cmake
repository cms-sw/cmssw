# TestHelpers.cmake
# 
# set of Convenience functions for unit testing with cmake
#
# License: BSD 3
#
# Copyright (c) 2016, Adrien Devresse
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 



##
# enable or disable detection of SLURM and MPIEXEC
option(AUTO_TEST_WITH_SLURM "automatically add srun as test prefix in a SLURM environment" TRUE)
option(AUTO_TEST_WITH_MPIEXEC "automatically add mpiexec as test prefix in a MPICH2/OpenMPI environment" TRUE)

###
##
## Basic SLURM support
## the prefix "srun" is added to any test in the environment
## For a slurm test execution, simply run "salloc [your_exec_parameters] ctest"
##    
##
if(AUTO_TEST_WITH_SLURM)
    if(NOT DEFINED SLURM_SRUN_COMMAND)
        find_program(SLURM_SRUN_COMMAND
                       NAMES "srun"
                       HINTS "${SLURM_ROOT}/bin")
    endif()
    
    if(SLURM_SRUN_COMMAND)
        set(TEST_EXEC_PREFIX_DEFAULT "${SLURM_SRUN_COMMAND}")
        set(TEST_MPI_EXEC_PREFIX_DEFAULT "${SLURM_SRUN_COMMAND}")
        set(TEST_MPI_EXEC_BIN_DEFAULT "${SLURM_SRUN_COMMAND}")
	set(TEST_WITH_SLURM ON)
        message(STATUS " - AUTO_TEST_WITH_SLURM with slurm cmd ${TEST_EXEC_PREFIX_DEFAULT} ")
        message(STATUS "  -- set test execution prefix to ${TEST_EXEC_PREFIX_DEFAULT} ")
        message(STATUS "  -- set MPI test execution prefix to ${TEST_MPI_EXEC_PREFIX_DEFAULT} ")
    endif()

endif()

###
## Basic MPIExec support, will just forward mpiexec as prefix
## 
if(AUTO_TEST_WITH_MPIEXEC AND NOT TEST_WITH_SLURM)

   if(NOT DEFINED MPIEXEC)
        find_program(MPIEXEC
                     NAMES "mpiexec"
                     HINTS "${MPI_ROOT}/bin")
   endif()


   if(MPIEXEC)
        set(TEST_MPI_EXEC_PREFIX_DEFAULT "${MPIEXEC}")
        set(TEST_MPI_EXEC_BIN_DEFAULT "${MPIEXEC}")
	set(TEST_WITH_MPIEXEC ON)
        message(STATUS " - AUTO_TEST_WITH_MPIEXEC cmd ${MPIEXEC} ")
        message(STATUS "  -- set MPI test execution prefix to ${TEST_MPI_EXEC_PREFIX_DEFAULT} ")

   endif()

endif()



###
##  MPI executor program path without arguments used for testing.
##  default: srun or mpiexec if found
##
set(TEST_MPI_EXEC_BIN "${TEST_MPI_EXEC_BIN_DEFAULT}" CACHE STRING "path of the MPI executor (mpiexec, mpirun) for test execution")



###
## Test execution prefix. Override this variable for any execution prefix required in clustered environment
## 
## To specify manually a command with argument, e.g -DTEST_EXEC_PREFIX="/var/empty/bin/srun;-n;-4" for a srun execution
## with 4 nodes
##
## default: srun if found
##
set(TEST_EXEC_PREFIX "${TEST_EXEC_PREFIX_DEFAULT}" CACHE STRING "prefix command for the test executions")



###
## Test execution prefix specific for MPI programs.
## 
## To specify manually a command with argument, use the cmake list syntax. e.g -DTEST_EXEC_PREFIX="/var/empty/bin/mpiexec;-n;-4" for an MPI execution
## with 4 nodes
##
## default: srun or mpiexec if found
##
set(TEST_MPI_EXEC_PREFIX "${TEST_MPI_EXEC_PREFIX_DEFAULT}" CACHE STRING "prefix command for the MPI test executions")







