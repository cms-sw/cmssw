import FWCore.ParameterSet.Config as cms

jetDQMParameters = cms.PSet(
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

    ptBin       = cms.int32(90),
    ptMax       = cms.double(100.0),
    ptMin       = cms.double(10.0),

    ptThreshold = cms.double(20.),
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
##ok i have to add here parameters which will be used only
##in the case of PFjets
    fillpfJIDPassFrac = cms.int32(0),
    ThisCHFMin = cms.double(-999.),
    ThisNHFMax = cms.double(999.),
    ThisCEFMax = cms.double(999.),
    ThisNEFMax = cms.double(999.),
    TightCHFMin = cms.double(0.0),
    TightNHFMax = cms.double(1.0),
    TightCEFMax = cms.double(1.0),
    TightNEFMax = cms.double(1.0),
    LooseCHFMin = cms.double(0.0),
    LooseNHFMax = cms.double(0.9),
    LooseCEFMax = cms.double(1.0),
    LooseNEFMax = cms.double(0.9)
)


cleanedJetDQMParameters = jetDQMParameters.clone(
    fillJIDPassFrac   = cms.int32(0),
    ptThreshold = cms.double(10.),
    n90HitsMin  = cms.int32(2),
    fHPDMax     = cms.double(0.98),
    resEMFMin   = cms.double(0.01),
    fillpfJIDPassFrac = cms.int32(1),
    ThisCHFMin = cms.double(0.0),
    ThisNHFMax = cms.double(0.9),
    ThisCEFMax = cms.double(1.0),
    ThisNEFMax = cms.double(0.9)
)
