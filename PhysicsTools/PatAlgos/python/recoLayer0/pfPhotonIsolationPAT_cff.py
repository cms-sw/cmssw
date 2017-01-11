import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.photonPFIsolationDepositsPAT_cff import *
from PhysicsTools.PatAlgos.recoLayer0.photonPFIsolationValuesPAT_cff import *

pfPhotonIsolationPATTask = cms.Task(
    photonPFIsolationDepositsPATTask,
    photonPFIsolationValuesPATTask
    )
pfPhotonIsolationPATSequence = cms.Sequence(pfPhotonIsolationPATTask)
