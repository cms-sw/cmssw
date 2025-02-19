import FWCore.ParameterSet.Config as cms

l1CaloGeometry = cms.ESProducer("L1CaloGeometryProd",

  numberGctEmJetPhiBins = cms.uint32(18),
  gctEmJetPhiBinOffset = cms.double(-0.5),
  numberGctEtSumPhiBins = cms.uint32(72),
  gctEtSumPhiBinOffset = cms.double(0.),
  numberGctHtSumPhiBins = cms.uint32(18),
  gctHtSumPhiBinOffset = cms.double(0.),
  numberGctCentralEtaBinsPerHalf = cms.uint32(7),
  numberGctForwardEtaBinsPerHalf = cms.uint32(4),
  etaSignBitOffset = cms.uint32(8),
  gctEtaBinBoundaries = cms.vdouble( 0.0000,
                0.3480,
                0.6950,
                1.0440,
                1.3920,
                1.7400,
                2.1720,
                3.0000,
                3.5000,
                4.0000,
                4.5000,
                5.0000 )
   )


