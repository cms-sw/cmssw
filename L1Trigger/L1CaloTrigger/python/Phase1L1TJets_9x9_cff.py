import FWCore.ParameterSet.Config as cms
from math import pi

from L1Trigger.L1CaloTrigger.Phase1L1TJetProducer_cfi import Phase1L1TJetProducer
from L1Trigger.L1CaloTrigger.Phase1L1TJetCalibrator_9x9Jets_cfi import Phase1L1TJetCalibrator as Phase1L1TJetCalibrator9x9
from L1Trigger.L1CaloTrigger.Phase1L1TJetSumsProducer_cfi import Phase1L1TJetSumsProducer

Phase1L1TJetProducer9x9 = Phase1L1TJetProducer.clone(
	  jetIEtaSize = 9,
	  jetIPhiSize = 9,
	  outputCollectionName = "UncalibratedPhase1L1TJetFromPfCandidates"
)

Phase1L1TJetCalibrator9x9.inputCollectionTag = ("Phase1L1TJetProducer9x9", "UncalibratedPhase1L1TJetFromPfCandidates", "")
Phase1L1TJetCalibrator9x9.outputCollectionName = ("Phase1L1TJetFromPfCandidates")

Phase1L1TJetSumsProducer9x9 = Phase1L1TJetSumsProducer.clone(
  inputJetCollectionTag = ("Phase1L1TJetCalibrator9x9", "Phase1L1TJetFromPfCandidates"),
)

Phase1L1TJetsSequence9x9 = cms.Sequence(
  Phase1L1TJetProducer9x9 +
  Phase1L1TJetCalibrator9x9 + 
  Phase1L1TJetSumsProducer9x9
)