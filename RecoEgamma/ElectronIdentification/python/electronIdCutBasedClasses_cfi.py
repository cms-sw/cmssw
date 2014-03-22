import FWCore.ParameterSet.Config as cms

eidCutBasedClasses = cms.EDFilter("EleIdCutBasedRef",

    filter = cms.bool(False),
    threshold = cms.double(0.5),

    src = cms.InputTag("gedGsfElectrons"),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    algorithm = cms.string('eIDCBClasses'),

    electronQuality = cms.string('loose'),

    useBremFraction = cms.vint32(0, 0, 0),
    useInvEMinusInvP = cms.vint32(0, 0, 0),
    useHoverE = cms.vint32(1, 1, 1),
    useSigmaEtaEta = cms.vint32(0, 1, 1),
    useSigmaPhiPhi = cms.vint32(0, 0, 0),
    useE9overE25 = cms.vint32(1, 1, 1),
    useEoverPOut = cms.vint32(1, 1, 1),
    useEoverPIn = cms.vint32(0, 0, 0),
    useDeltaPhiIn = cms.vint32(1, 1, 1),
    useDeltaPhiOut = cms.vint32(0, 1, 1),
    useDeltaEtaIn = cms.vint32(1, 1, 1),
    acceptCracks = cms.vint32(1, 1, 1),

    looseEleIDCuts = cms.PSet(
        invEMinusInvP = cms.vdouble(0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
        EoverPInMin = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        EoverPOutMin = cms.vdouble(0.7, 1.7, 0.9, 0.6, 0.7, 1.7, 0.9, 0.6, 0.5),
        sigmaEtaEtaMin = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        EoverPOutMax = cms.vdouble(2.5, 999.0, 2.2, 999.0, 2.5, 999.0, 2.2, 999.0, 999.0),
        EoverPInMax = cms.vdouble(999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0),
        deltaPhiOut = cms.vdouble(0.011, 999.0, 999.0, 999.0, 0.02, 999.0, 999.0, 999.0, 999.0),
        sigmaEtaEtaMax = cms.vdouble(999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0),
        deltaPhiIn = cms.vdouble(0.02, 0.06, 0.06, 0.08, 0.02, 0.06, 0.06, 0.08, 0.08),
        HoverE = cms.vdouble(0.06, 0.06, 0.07, 0.08, 0.06, 0.06, 0.07, 0.08, 0.12),
        sigmaPhiPhiMin = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        bremFraction = cms.vdouble(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        deltaEtaIn = cms.vdouble(0.005, 0.008, 0.008, 0.009, 0.005, 0.008, 0.008, 0.009, 0.009),
        E9overE25 = cms.vdouble(0.8, 0.7, 0.7, 0.5, 0.8, 0.8, 0.8, 0.8, 0.5),
        sigmaPhiPhiMax = cms.vdouble(999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0)
    ),
    tightEleIDCuts = cms.PSet(
        invEMinusInvP = cms.vdouble(0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
        EoverPInMin = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        EoverPOutMin = cms.vdouble(0.6, 0.75, 0.75, 0.75, 0.5, 0.8, 0.5, 0.8, 0.75),
        sigmaEtaEtaMin = cms.vdouble(0.005, 0.005, 0.005, 0.005, 0.008, 0.008, 0.008, 0.008, 0.005),
        EoverPOutMax = cms.vdouble(999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0),
        EoverPInMax = cms.vdouble(999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0),
        deltaPhiOut = cms.vdouble(0.02, 999.0, 0.02, 999.0, 0.02, 999.0, 0.02, 999.0, 999.0),
        sigmaEtaEtaMax = cms.vdouble(0.011, 0.011, 0.011, 0.011, 0.03, 0.03, 0.03, 0.022, 0.011),
        deltaPhiIn = cms.vdouble(0.02, 0.03, 0.02, 0.04, 0.04, 0.04, 0.04, 0.05, 0.04),
        HoverE = cms.vdouble(0.05, 0.05, 0.05, 0.05, 0.07, 0.07, 0.07, 0.07, 0.05),
        sigmaPhiPhiMin = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        bremFraction = cms.vdouble(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        deltaEtaIn = cms.vdouble(0.004, 0.004, 0.004, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005),
        E9overE25 = cms.vdouble(0.8, 0.65, 0.75, 0.65, 0.8, 0.7, 0.7, 0.65, 0.65),
        sigmaPhiPhiMax = cms.vdouble(999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0)
    ),
    mediumEleIDCuts = cms.PSet(
        invEMinusInvP = cms.vdouble(0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
        EoverPInMin = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        EoverPOutMin = cms.vdouble(0.6, 1.8, 1.0, 0.75, 0.6, 1.5, 1.0, 0.8, 1.0),
        sigmaEtaEtaMin = cms.vdouble(0.005, 0.005, 0.005, 0.005, 0.008, 0.008, 0.008, 0.0, 0.005),
        EoverPOutMax = cms.vdouble(2.5, 999.0, 999.0, 999.0, 2.0, 999.0, 999.0, 999.0, 999.0),
        EoverPInMax = cms.vdouble(999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0),
        deltaPhiOut = cms.vdouble(0.011, 999.0, 999.0, 999.0, 0.02, 999.0, 999.0, 999.0, 999.0),
        sigmaEtaEtaMax = cms.vdouble(0.011, 0.011, 0.011, 0.011, 0.022, 0.022, 0.022, 0.3, 0.011),
        deltaPhiIn = cms.vdouble(0.04, 0.07, 0.04, 0.08, 0.06, 0.07, 0.06, 0.07, 0.08),
        HoverE = cms.vdouble(0.06, 0.05, 0.06, 0.14, 0.1, 0.1, 0.1, 0.12, 0.14),
        sigmaPhiPhiMin = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        bremFraction = cms.vdouble(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
        deltaEtaIn = cms.vdouble(0.004, 0.006, 0.005, 0.007, 0.007, 0.008, 0.007, 0.008, 0.007),
        E9overE25 = cms.vdouble(0.7, 0.75, 0.8, 0.0, 0.85, 0.75, 0.8, 0.0, 0.0),
        sigmaPhiPhiMax = cms.vdouble(999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0)
    )
)


