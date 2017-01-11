import FWCore.ParameterSet.Config as cms

# iso deposits and isolation values, defined by the Muon POG

from PhysicsTools.PatAlgos.recoLayer0.muonPFIsolationDepositsPAT_cff import *
from PhysicsTools.PatAlgos.recoLayer0.muonPFIsolationValuesPAT_cff import *

muonPFIsolationPATTask = cms.Task(
    muonPFIsolationDepositsPATTask,
    muonPFIsolationValuesPATTask
)
muonPFIsolationPATSequence = cms.Sequence(muonPFIsolationPATTask)
