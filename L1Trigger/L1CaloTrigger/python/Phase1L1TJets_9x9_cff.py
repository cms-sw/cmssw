import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.l1tPhase1JetProducer_cfi import l1tPhase1JetProducer
from L1Trigger.L1CaloTrigger.l1tPhase1JetCalibrator9_cfi import l1tPhase1JetCalibrator9
from L1Trigger.L1CaloTrigger.l1tPhase1JetSumsProducer_cfi import l1tPhase1JetSumsProducer

l1tPhase1JetProducer9x9 = l1tPhase1JetProducer.clone(
	  jetIEtaSize = 9,
	  jetIPhiSize = 9,
	  outputCollectionName = "UncalibratedPhase1L1TJetFromPfCandidates"
)

l1tPhase1JetCalibrator9x9 = l1tPhase1JetCalibrator9.clone(
	  inputCollectionTag = ("l1tPhase1JetProducer9x9", "UncalibratedPhase1L1TJetFromPfCandidates", ""),
	  outputCollectionName = "Phase1L1TJetFromPfCandidates"
)

l1tPhase1JetSumsProducer9x9 = l1tPhase1JetSumsProducer.clone(
  inputJetCollectionTag = ("l1tPhase1JetCalibrator9x9", "Phase1L1TJetFromPfCandidates"),
)

L1TPhase1JetsSequence9x9 = cms.Sequence(
  l1tPhase1JetProducer9x9 +
  l1tPhase1JetCalibrator9x9 + 
  l1tPhase1JetSumsProducer9x9
)
