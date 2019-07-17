import FWCore.ParameterSet.Config as cms
from math import pi

caloEtaSegmentation = cms.vdouble(
  -5.184, -4.883, -4.709, -4.532, -4.357, -4.184, -4.006, -3.833, -3.657, -3.482, -3.307, -3.132, -3,
  -2.919, -2.839, -2.759, -2.679, -2.599, -2.519, -2.439, -2.359, -2.279, -2.199, -2.119, -2.039, -1.959,
  -1.879, -1.799, -1.719, -1.639, -1.559, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -0.957, -0.87,
  -0.783, -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087, 0, 0.087, 0.174, 0.261, 0.348,
  0.435, 0.522, 0.609, 0.696, 0.783, 0.87, 0.957, 1.044, 1.131, 1.218, 1.305, 1.392, 1.479, 1.559, 1.639,
  1.719, 1.799, 1.879, 1.959, 2.039, 2.119, 2.199, 2.279, 2.359, 2.439, 2.519, 2.599, 2.679, 2.759, 2.839,
  2.919, 3, 3.132, 3.307, 3.482, 3.657, 3.833, 4.006, 4.184, 4.357, 4.532, 4.709, 4.883, 5.184
)

Phase1L1TJetProducer = cms.EDProducer('Phase1L1TJetProducer',
  inputCollectionTag = cms.InputTag("l1pfCandidates", "Puppi", "IN"),
  etaBinning = caloEtaSegmentation,
  nBinsPhi = cms.uint32(72),
  phiLow = cms.double(-pi),
  phiUp = cms.double(pi),
  jetIEtaSize = cms.uint32(5),
  jetIPhiSize = cms.uint32(5),
  seedPtThreshold = cms.double(5), # GeV
  puSubtraction = cms.bool(False),
  outputCollectionName = cms.string("UncalibratedPhase1L1TJetFromPfCandidates"),
  vetoZeroPt = cms.bool(True)
)
