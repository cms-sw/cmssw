import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.Phase1L1TJetProducer_cfi import l1tPhase1JetProducer
from L1Trigger.L1CaloTrigger.Phase1L1TJetCalibrator_9x9trimmedJets_cfi import l1tPhase1JetCalibrator
from L1Trigger.L1CaloTrigger.Phase1L1TJetSumsProducer_cfi import l1tPhase1JetSumsProducer

l1tPhase1JetProducer9x9trimmed = l1tPhase1JetProducer.clone(
	  jetIEtaSize = 9,
	  jetIPhiSize = 9,
	  trimmedGrid = True,
	  outputCollectionName = "UncalibratedPhase1L1TJetFromPfCandidates"
)

l1tPhase1JetCalibrator9x9trimmed = l1tPhase1JetCalibrator.clone(
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
