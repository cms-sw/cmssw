import FWCore.ParameterSet.Config as cms

hbhereco = cms.EDProducer(
    'HBHEIsolatedNoiseReflagger',

    debug = cms.untracked.bool(False),

    hbheInput = cms.InputTag('hbheprereco'),
    ebInput = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    eeInput = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    trackExtrapolationInput = cms.InputTag('trackExtrapolator'),

    # isolation cuts
    LooseHcalIsol = cms.double(0.08),
    LooseEcalIsol = cms.double(0.08),
    LooseTrackIsol = cms.double(0.10),
    TightHcalIsol = cms.double(0.04),
    TightEcalIsol = cms.double(0.04),
    TightTrackIsol = cms.double(0.05),

    LooseRBXEne1 = cms.double(10.0),
    LooseRBXEne2 = cms.double(300.0),
    LooseRBXHits1 = cms.int32(14),
    LooseRBXHits2 = cms.int32(10),
    TightRBXEne1 = cms.double(10.0),
    TightRBXEne2 = cms.double(50.0),
    TightRBXHits1 = cms.int32(12),
    TightRBXHits2 = cms.int32(10),

    LooseHPDEne1 = cms.double(10.0),
    LooseHPDEne2 = cms.double(80.0),
    LooseHPDHits1 = cms.int32(6),
    LooseHPDHits2 = cms.int32(3),
    TightHPDEne1 = cms.double(10.0),
    TightHPDEne2 = cms.double(25.0),
    TightHPDHits1 = cms.int32(5),
    TightHPDHits2 = cms.int32(3),

    LooseDiHitEne = cms.double(50.0),
    TightDiHitEne = cms.double(15.0),
    LooseMonoHitEne = cms.double(35.0),
    TightMonoHitEne = cms.double(15.0),

    # used by the object validator
    HBThreshold = cms.double(0.7),
    HESThreshold = cms.double(0.8),
    HEDThreshold = cms.double(0.8),
    EBThreshold = cms.double(0.07),
    EEThreshold = cms.double(0.3),
    HcalAcceptSeverityLevel = cms.uint32(9),
    EcalAcceptSeverityLevel = cms.uint32(3),
    UseHcalRecoveredHits = cms.bool(True),
    UseEcalRecoveredHits = cms.bool(False),
    MinValidTrackPt = cms.double(0.3),
    MinValidTrackPtBarrel = cms.double(0.9),
    MinValidTrackNHits = cms.int32(5),

 )
