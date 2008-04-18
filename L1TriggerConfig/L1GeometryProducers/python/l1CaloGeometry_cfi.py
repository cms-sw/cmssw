import FWCore.ParameterSet.Config as cms

l1CaloGeometry = cms.ESProducer("L1CaloGeometryProd",
    gctEmJetPhiBinOffset = cms.double(-0.5),
    numberGctEmJetPhiBins = cms.uint32(18),
    numberGctEtSumPhiBins = cms.uint32(72),
    gctEtaBinBoundaries = cms.vdouble(0.0, 0.348, 0.695, 1.044, 1.392, 
        1.74, 2.172, 3.0, 3.5, 4.0, 
        4.5, 5.0),
    numberGctCentralEtaBinsPerHalf = cms.uint32(7),
    gctEtSumPhiBinOffset = cms.double(0.0),
    numberGctForwardEtaBinsPerHalf = cms.uint32(4),
    etaSignBitOffset = cms.uint32(8)
)



