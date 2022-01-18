import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import *

# Customization to run the CPU vs GPU comparison task if the job runs on a GPU enabled machine
from Configuration.ProcessModifiers.gpu_cff import gpu
from DQM.EcalMonitorTasks.GpuTask_cfi import ecalGpuTask

gpu.toModify(ecalGpuTask.params, runGpuTask = cms.untracked.bool(True))
gpu.toModify(ecalMonitorTask.workers, func = lambda workers: workers.append("GpuTask"))
gpu.toModify(ecalMonitorTask, workerParameters = dict(GpuTask = ecalGpuTask))
