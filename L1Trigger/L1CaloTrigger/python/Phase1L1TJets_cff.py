import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.l1tPhase1JetProducer_cfi import l1tPhase1JetProducer
from L1Trigger.L1CaloTrigger.l1tPhase1JetCalibrator_cfi import l1tPhase1JetCalibrator
from L1Trigger.L1CaloTrigger.l1tPhase1JetSumsProducer_cfi import l1tPhase1JetSumsProducer

L1TPhase1JetsSequence = cms.Sequence(
  l1tPhase1JetProducer +
  l1tPhase1JetCalibrator +
  l1tPhase1JetSumsProducer
)
