import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import *

# Customization to run the CPU vs GPU comparison task if the job runs on a GPU enabled machine
from Configuration.ProcessModifiers.gpuValidation_cff import gpuValidation
from DQM.EcalMonitorTasks.GpuTask_cfi import ecalGpuTask

gpuValidation.toModify(ecalGpuTask.params, runGpuTask = cms.untracked.bool(True))
gpuValidation.toModify(ecalMonitorTask.workers, func = lambda workers: workers.append("GpuTask"))
gpuValidation.toModify(ecalMonitorTask, workerParameters = dict(GpuTask = ecalGpuTask))
