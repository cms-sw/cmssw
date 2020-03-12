import FWCore.ParameterSet.Config as cms

from Calibration.IsolatedParticles.isoTrigDefault_cfi import isoTrigDefault as _isoTrigDefault

isoTrigHB = _isoTrigDefault.clone()
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(isoTrigHB, stageL1Trigger = 2)

isoTrigHE = _isoTrigDefault.clone(
  Triggers = cms.untracked.vstring('HLT_IsoTrackHE'))
stage2L1Trigger.toModify(isoTrigHE, stageL1Trigger = 2)
