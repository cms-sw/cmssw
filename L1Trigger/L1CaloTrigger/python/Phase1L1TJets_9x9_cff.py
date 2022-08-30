import FWCore.ParameterSet.Config as cms
from math import pi

from L1Trigger.L1CaloTrigger.Phase1L1TJetProducer_cfi import l1tPhase1JetProducer
from L1Trigger.L1CaloTrigger.Phase1L1TJetCalibrator_9x9Jets_cfi import l1tPhase1JetCalibrator as l1tPhase1JetCalibrator9x9
from L1Trigger.L1CaloTrigger.Phase1L1TJetSumsProducer_cfi import l1tPhase1JetSumsProducer

l1tPhase1JetProducer9x9 = l1tPhase1JetProducer.clone(
	  jetIEtaSize = 9,
	  jetIPhiSize = 9,
	  outputCollectionName = "UncalibratedPhase1L1TJetFromPfCandidates"
)

l1tPhase1JetCalibrator9x9.inputCollectionTag = ("l1tPhase1JetProducer9x9", "UncalibratedPhase1L1TJetFromPfCandidates", "")
l1tPhase1JetCalibrator9x9.outputCollectionName = ("Phase1L1TJetFromPfCandidates")

l1tPhase1JetSumsProducer9x9 = l1tPhase1JetSumsProducer.clone(
  inputJetCollectionTag = ("Phase1L1TJetCalibrator9x9", "Phase1L1TJetFromPfCandidates"),
)

L1TPhase1JetsSequence9x9 = cms.Sequence(
  l1tPhase1JetProducer9x9 +
  l1tPhase1JetCalibrator9x9 + 
  l1tPhase1JetSumsProducer9x9
)
