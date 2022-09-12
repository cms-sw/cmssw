import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.l1tPhase1JetProducer_cfi import l1tPhase1JetProducer
from L1Trigger.L1CaloTrigger.l1tPhase1JetCalibrator_9x9trimmed_cfi import l1tPhase1JetCalibrator_9x9trimmed
from L1Trigger.L1CaloTrigger.l1tPhase1JetSumsProducer_cfi import l1tPhase1JetSumsProducer

l1tPhase1JetProducer9x9trimmed = l1tPhase1JetProducer.clone(
	  jetIEtaSize = 9,
	  jetIPhiSize = 9,
	  trimmedGrid = True,
	  outputCollectionName = "UncalibratedPhase1L1TJetFromPfCandidates"
)

l1tPhase1JetCalibrator9x9trimmed = l1tPhase1JetCalibrator_9x9trimmed.clone(
	  inputCollectionTag = ("l1tPhase1JetProducer9x9trimmed", "UncalibratedPhase1L1TJetFromPfCandidates", ""),
	  outputCollectionName = "Phase1L1TJetFromPfCandidates"
)

l1tPhase1JetSumsProducer9x9trimmed = l1tPhase1JetSumsProducer.clone(
  inputJetCollectionTag = ("l1tPhase1JetCalibrator9x9trimmed", "Phase1L1TJetFromPfCandidates"),
)

L1TPhase1JetsSequence9x9trimmed = cms.Sequence(
  l1tPhase1JetProducer9x9trimmed +
  l1tPhase1JetCalibrator9x9trimmed + 
  l1tPhase1JetSumsProducer9x9trimmed
)
