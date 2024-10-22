import FWCore.ParameterSet.Config as cms
from math import pi

from L1Trigger.L1CaloTrigger.Phase1L1TJets_sincosLUT_cff import sinPhi, cosPhi

l1tPhase1JetSumsProducer = cms.EDProducer('Phase1L1TJetSumsProducer',
  inputJetCollectionTag = cms.InputTag("l1tPhase1JetCalibrator", "Phase1L1TJetFromPfCandidates"),
  nBinsPhi = cms.uint32(72),
  phiLow = cms.double(-pi),
  phiUp = cms.double(pi),
  sinPhi = sinPhi,
  cosPhi = cosPhi,
  htPtThreshold = cms.double(30),
  htAbsEtaCut = cms.double(2.4),
  mhtPtThreshold = cms.double(30),
  mhtAbsEtaCut = cms.double(2.4),
  ptlsb = cms.double(0.25),
  outputCollectionName = cms.string("Sums"),
)
