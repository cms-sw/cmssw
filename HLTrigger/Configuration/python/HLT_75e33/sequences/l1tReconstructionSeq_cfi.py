import FWCore.ParameterSet.Config as cms

from ..modules.l1tSlwPFPuppiJets_cfi import *
from ..modules.l1tSlwPFPuppiJetsCorrected_cfi import *

l1tReconstructionSeq = cms.Sequence(l1tSlwPFPuppiJets+l1tSlwPFPuppiJetsCorrected)
