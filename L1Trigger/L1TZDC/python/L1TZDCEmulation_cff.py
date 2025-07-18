import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TZDC.l1tZDCEtSums_cfi import l1tZDCEtSums

L1TZDCEmulationTask = cms.Task(
    l1tZDCEtSums
)
