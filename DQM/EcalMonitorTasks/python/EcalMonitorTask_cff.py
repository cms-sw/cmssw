import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import *

# Customization to run the CPU vs GPU comparison task if the job runs on a GPU enabled machine
from Configuration.ProcessModifiers.gpuValidationEcal_cff import gpuValidationEcal
from DQM.EcalMonitorTasks.ecalGpuTask_cfi import ecalGpuTask

# Input tags used for offline DQM RECO, enables SwitchProducerCUDA to generate these collections
gpuValidationEcal.toModify(ecalDQMCollectionTags, EBCpuDigi = cms.untracked.InputTag("ecalDigis@cpu", "ebDigis"))
gpuValidationEcal.toModify(ecalDQMCollectionTags, EECpuDigi = cms.untracked.InputTag("ecalDigis@cpu", "eeDigis"))
gpuValidationEcal.toModify(ecalDQMCollectionTags, EBGpuDigi = cms.untracked.InputTag("ecalDigis@cuda", "ebDigis"))
gpuValidationEcal.toModify(ecalDQMCollectionTags, EEGpuDigi = cms.untracked.InputTag("ecalDigis@cuda", "eeDigis"))
gpuValidationEcal.toModify(ecalDQMCollectionTags, EBCpuUncalibRecHit = cms.untracked.InputTag("ecalMultiFitUncalibRecHit@cpu", "EcalUncalibRecHitsEB"))
gpuValidationEcal.toModify(ecalDQMCollectionTags, EECpuUncalibRecHit = cms.untracked.InputTag("ecalMultiFitUncalibRecHit@cpu", "EcalUncalibRecHitsEE"))
gpuValidationEcal.toModify(ecalDQMCollectionTags, EBGpuUncalibRecHit = cms.untracked.InputTag("ecalMultiFitUncalibRecHit@cuda", "EcalUncalibRecHitsEB"))
gpuValidationEcal.toModify(ecalDQMCollectionTags, EEGpuUncalibRecHit = cms.untracked.InputTag("ecalMultiFitUncalibRecHit@cuda", "EcalUncalibRecHitsEE"))
gpuValidationEcal.toModify(ecalDQMCollectionTags, EBCpuRecHit = cms.untracked.InputTag("ecalRecHit@cpu", "EcalRecHitsEB"))
gpuValidationEcal.toModify(ecalDQMCollectionTags, EECpuRecHit = cms.untracked.InputTag("ecalRecHit@cpu", "EcalRecHitsEE"))
gpuValidationEcal.toModify(ecalDQMCollectionTags, EBGpuRecHit = cms.untracked.InputTag("ecalRecHit@cuda", "EcalRecHitsEB"))
gpuValidationEcal.toModify(ecalDQMCollectionTags, EEGpuRecHit = cms.untracked.InputTag("ecalRecHit@cuda", "EcalRecHitsEE"))
