import FWCore.ParameterSet.Config as cms
jetDQMParameters = cms.PSet(
    verbose     = cms.int32(0),
    eBin        = cms.int32(100),
    eMax        = cms.double(500.0),
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
    n90HitsMin  = cms.int32(-1),
    fHPDMax     = cms.double(1.),
    resEMFMin   = cms.double(0.),
    fillJIDPassFrac   = cms.int32(1),

    n90HitsMinLoose  = cms.int32(1),
    fHPDMaxLoose     = cms.double(0.98),
    resEMFMinLoose   = cms.double(0.01),    
    n90HitsMinTight  = cms.int32(4),
    fHPDMaxTight     = cms.double(0.98),
    resEMFMinTight   = cms.double(0.01),

    sigmaEtaMinTight   = cms.double(0.01),    
    sigmaPhiMinTight   = cms.double(0.01),
    
    makedijetselection  = cms.int32(0),

    JetIDParams  = cms.PSet(
        useRecHits      = cms.bool(True),
        hbheRecHitsColl = cms.InputTag("hbhereco"),
        hoRecHitsColl   = cms.InputTag("horeco"),
        hfRecHitsColl   = cms.InputTag("hfreco"),
        ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
        eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
    ),

    #PF specific cleaning values
    ThisCHFMin = cms.double(-999.),
    ThisNHFMax = cms.double(999.),
    ThisCEFMax = cms.double(999.),
    ThisNEFMax = cms.double(999.),
    TightCHFMin = cms.double(0.0),
    TightNHFMax = cms.double(0.9),
    TightCEFMax = cms.double(1.0),
    TightNEFMax = cms.double(0.9),
    LooseCHFMin = cms.double(0.0),
    LooseNHFMax = cms.double(1.0),
    LooseCEFMax = cms.double(1.0),
    LooseNEFMax = cms.double(1.0)
)



