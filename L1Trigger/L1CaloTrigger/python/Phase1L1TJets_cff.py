import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.Phase1L1TJetProducer_cfi import Phase1L1TJetProducer
from L1Trigger.L1CaloTrigger.Phase1L1TJetCalibrator_cfi import Phase1L1TJetCalibrator

Phase1L1TJetsSequence = cms.Sequence(
  Phase1L1TJetProducer +
  Phase1L1TJetCalibrator
)
