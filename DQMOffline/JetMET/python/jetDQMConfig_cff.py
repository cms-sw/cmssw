import FWCore.ParameterSet.Config as cms
jetDQMParameters = cms.PSet(
    verbose     = cms.int32(0),
    eBin        = cms.int32(100),
    eMax        = cms.double(1000.0),
    eMin        = cms.double(0.0),

    etaBin      = cms.int32(100),
    etaMax      = cms.double(5.0),
    etaMin      = cms.double(-5.0),

    pBin        = cms.int32(100),
    pMax        = cms.double(500.0),
    pMin        = cms.double(0.0),

    phiBin      = cms.int32(70),
    phiMax      = cms.double(3.2),
    phiMin      = cms.double(-3.2),

    ptBin       = cms.int32(100),
    ptMax       = cms.double(500.0),
    ptMin       = cms.double(20.0),

    pVBin       = cms.int32(100),
    pVMax       = cms.double(100.0),
    pVMin       = cms.double(0.0),

    ptThreshold     = cms.double(20.),
    ptThresholdUnc  = cms.double(17.5),
    asymmetryThirdJetCut = cms.double(30),
    balanceThirdJetCut   = cms.double(0.2),
    fillJIDPassFrac   = cms.int32(1),
)



