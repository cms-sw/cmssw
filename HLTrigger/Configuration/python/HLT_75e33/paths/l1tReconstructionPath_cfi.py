import FWCore.ParameterSet.Config as cms

from ..tasks.l1tReconstructionTask_cfi import *

l1tReconstructionPath = cms.Path(l1tReconstructionTask)
