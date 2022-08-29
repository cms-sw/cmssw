import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.Phase1L1TJetProducer_cfi import l1tPhase1JetProducer
from L1Trigger.L1CaloTrigger.Phase1L1TJetCalibrator_cfi import l1tPhase1JetCalibrator
from L1Trigger.L1CaloTrigger.Phase1L1TJetSumsProducer_cfi import l1tPhase1JetSumsProducer

l1tPhase1JetsSequence = cms.Sequence(
  l1tPhase1JetProducer +
  l1tPhase1JetCalibrator +
  l1tPhase1JetSumsProducer
)
