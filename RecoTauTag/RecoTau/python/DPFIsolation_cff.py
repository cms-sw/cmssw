
import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.DPFIsolation_cfi import *

DPFIsolationTask = cms.Task(DPFIsolation)
DPFIsolationSeq = cms.Sequence(DPFIsolationTask)
