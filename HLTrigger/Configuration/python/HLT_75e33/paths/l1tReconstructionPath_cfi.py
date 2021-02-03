import FWCore.ParameterSet.Config as cms

from ..sequences.l1tReconstructionSeq_cfi import *

l1tReconstructionPath = cms.Path(l1tReconstructionSeq)
