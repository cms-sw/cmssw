from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms
from math import pi

from L1Trigger.L1CaloTrigger.Phase1L1TJets_sincosLUT_cff import sinPhi, cosPhi

caloEtaSegmentation = cms.vdouble(
  -5.0, -4.917, -4.833, -4.75, -4.667, -4.583, -4.5, -4.417, -4.333, -4.25, 
  -4.167, -4.083, -4.0, -3.917, -3.833, -3.75, -3.667, -3.583, -3.5, -3.417, 
  -3.333, -3.25, -3.167, -3.083, -3.0, -2.917, -2.833, -2.75, -2.667, -2.583, 
  -2.5, -2.417, -2.333, -2.25, -2.167, -2.083, -2.0, -1.917, -1.833, -1.75, 
  -1.667, -1.583, -1.5, -1.417, -1.333, -1.25, -1.167, -1.083, -1.0, -0.917, 
  -0.833, -0.75, -0.667, -0.583, -0.5, -0.417, -0.333, -0.25, -0.167, -0.083, 
  0.0, 0.083, 0.167, 0.25, 0.333, 0.417, 0.5, 0.583, 0.667, 0.75, 0.833, 0.917, 
  1.0, 1.083, 1.167, 1.25, 1.333, 1.417, 1.5, 1.583, 1.667, 1.75, 1.833, 1.917, 
  2.0, 2.083, 2.167, 2.25, 2.333, 2.417, 2.5, 2.583, 2.667, 2.75, 2.833, 2.917, 
  3.0, 3.083, 3.167, 3.25, 3.333, 3.417, 3.5, 3.583, 3.667, 3.75, 3.833, 3.917, 
  4.0, 4.083, 4.167, 4.25, 4.333, 4.417, 4.5, 4.583, 4.667, 4.75, 4.833, 4.917, 5.0)

l1tPhase1JetProducer = cms.EDProducer('Phase1L1TJetProducer',
  inputCollectionTag = cms.InputTag("l1tLayer1", "Puppi"),
  etaBinning = caloEtaSegmentation,
  nBinsPhi = cms.uint32(72),
  phiLow = cms.double(-pi),
  phiUp = cms.double(pi),
  jetIEtaSize = cms.uint32(7),
  jetIPhiSize = cms.uint32(7),
  trimmedGrid = cms.bool(False),
  seedPtThreshold = cms.double(5), # GeV
  puSubtraction = cms.bool(False),
  ptlsb = cms.double(0.25),
  philsb = cms.double(0.0043633231),
  etalsb = cms.double(0.0043633231),
  outputCollectionName = cms.string("UncalibratedPhase1L1TJetFromPfCandidates"),
  vetoZeroPt = cms.bool(True),
  etaRegions = cms.vdouble( -5., -4.5, -4., -3.5, -3., -2.5, -1.5, -0.75, 0, 0.75, 1.5, 2.5, 3., 3.5, 4., 4.5, 5. ),
  phiRegions = cms.vdouble( -3.5, -2.8, -2.1, -1.4, -0.7, 0, 0.7, 1.4, 2.1, 2.8, 3.5 ),#, 4.2, 4.9, 5.6, 6.3 ),
  maxInputsPerRegion = cms.uint32( 18 ),
  sinPhi = sinPhi,
  cosPhi = cosPhi,
  metAbsEtaCut = cms.double(3),
  metHFAbsEtaCut = cms.double(5),
)
