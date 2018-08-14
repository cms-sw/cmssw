import FWCore.ParameterSet.Config as cms

from Calibration.IsolatedParticles.isoTrigHB_cfi import *

isoTrigHE = isoTrigHB.clone(
  Triggers = cms.untracked.vstring('HLT_IsoTrackHE'))
