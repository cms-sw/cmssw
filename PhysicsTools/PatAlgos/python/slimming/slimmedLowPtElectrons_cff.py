import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.slimmedLowPtElectrons_cfi import slimmedLowPtElectrons
from PhysicsTools.PatAlgos.slimming.lowPtGsfLinks_cfi import lowPtGsfLinks

# Task
slimmedLowPtElectronsTask = cms.Task(
    lowPtGsfLinks,
    slimmedLowPtElectrons,
)
