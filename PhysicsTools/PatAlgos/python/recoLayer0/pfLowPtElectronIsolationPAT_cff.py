import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.lowPtElectronPFIsolationDepositsPAT_cff import *
from PhysicsTools.PatAlgos.recoLayer0.lowPtElectronPFIsolationValuesPAT_cff import *

pfLowPtElectronIsolationPATTask = cms.Task(
    lowPtElectronPFIsolationDepositsPATTask,
    lowPtElectronPFIsolationValuesPATTask
    )
pfKLowPtElectronIsolationPATSequence = cms.Sequence(pfLowPtElectronIsolationPATTask)
