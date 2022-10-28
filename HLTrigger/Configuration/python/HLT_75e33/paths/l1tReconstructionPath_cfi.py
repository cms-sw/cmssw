import FWCore.ParameterSet.Config as cms

from ..tasks.l1tReconstructionTask_cfi import *

L1TReconstructionPath = cms.Path(
    L1TReconstructionTask
)
