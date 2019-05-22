
import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.DeepTauId_cfi import *

deepTauIdTask = cms.Task(deepTauIdraw)
deepTauIdSeq = cms.Sequence(deepTauIdTask)
