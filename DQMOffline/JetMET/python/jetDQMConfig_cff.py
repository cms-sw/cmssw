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

    ptBin       = cms.int32(100),
    ptMax       = cms.double(50.0),
    ptMin       = cms.double(0.0),

    ptThreshold = cms.double(3.),
    n90HitsMin  = cms.int32(-1),
    fHPDMax     = cms.double(1.),
    resEMFMin   = cms.double(0.),
    iscleaned   = cms.int32(0),

    makedijetselection  = cms.int32(0),

    jIDeffptBins = cms.int32(5),        
    JetIDParams  = cms.PSet(
        useRecHits      = cms.bool(True),
        hbheRecHitsColl = cms.InputTag("hbhereco"),
        hoRecHitsColl   = cms.InputTag("horeco"),
        hfRecHitsColl   = cms.InputTag("hfreco"),
        ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
        eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
    )
)


cleanedJetDQMParameters = jetDQMParameters.clone(
    iscleaned   = cms.int32(1),
    ptThreshold = cms.double(10.),
    n90HitsMin  = cms.int32(2),
    fHPDMax     = cms.double(0.98),
    resEMFMin   = cms.double(0.01)
)
