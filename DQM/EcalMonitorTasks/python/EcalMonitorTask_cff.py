import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import *

# Customization to run the CPU vs GPU comparison task if the job runs on a GPU enabled machine
from Configuration.ProcessModifiers.gpuValidationEcal_cff import gpuValidationEcal
from Configuration.ProcessModifiers.alpakaValidationEcal_cff import alpakaValidationEcal
from DQM.EcalMonitorTasks.ecalGpuTask_cfi import ecalGpuTask

gpuValidationEcal.toModify(ecalGpuTask.params, runGpuTask = True)
gpuValidationEcal.toModify(ecalMonitorTask.workers, func = lambda workers: workers.append("GpuTask"))
gpuValidationEcal.toModify(ecalMonitorTask, workerParameters = dict(GpuTask = ecalGpuTask))

alpakaValidationEcal.toModify(ecalGpuTask.params, runGpuTask = True)
alpakaValidationEcal.toModify(ecalMonitorTask.workers, func = lambda workers: workers.append("GpuTask"))
alpakaValidationEcal.toModify(ecalMonitorTask, workerParameters = dict(GpuTask = ecalGpuTask))

# Skip consuming and running over the EcalRawData collection for all GPU WFs
# This is to be used as long as the GPU unpacker unpacks a dummy EcalRawData collection
from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toModify(ecalMonitorTask.skipCollections, func = lambda skipCollections: skipCollections.append("EcalRawData"))

from Configuration.ProcessModifiers.alpaka_cff import alpaka
alpaka.toModify(ecalMonitorTask.skipCollections, func = lambda skipCollections: skipCollections.append("EcalRawData"))
