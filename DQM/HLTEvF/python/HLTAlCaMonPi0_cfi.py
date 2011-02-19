import FWCore.ParameterSet.Config as cms

EcalPi0Mon = cms.EDAnalyzer("HLTAlCaMonPi0",
    prescaleFactor = cms.untracked.int32(1),
    FolderName = cms.untracked.string('HLT/AlCaEcalPi0'),

    AlCaStreamEBpi0Tag = cms.untracked.InputTag("hltAlCaPi0RecHitsFilter","pi0EcalRecHitsEB"),
    AlCaStreamEEpi0Tag = cms.untracked.InputTag("hltAlCaPi0RecHitsFilter","pi0EcalRecHitsEE"),
    AlCaStreamEBetaTag = cms.untracked.InputTag("hltAlCaEtaRecHitsFilter","etaEcalRecHitsEB"),
    AlCaStreamEEetaTag = cms.untracked.InputTag("hltAlCaEtaRecHitsFilter","etaEcalRecHitsEE"),

    isMonEEpi0 = cms.untracked.bool(True),
    isMonEBpi0 = cms.untracked.bool(True),
    isMonEEeta = cms.untracked.bool(True),
    isMonEBeta = cms.untracked.bool(True),


    SaveToFile = cms.untracked.bool(False),
    FileName = cms.untracked.string('MonitorAlCaEcalPi0.root'),

    clusSeedThr = cms.double( 0.5 ),
    clusSeedThrEndCap = cms.double( 1.0 ),
    clusEtaSize = cms.int32( 3 ),
    clusPhiSize = cms.int32( 3 ),
    seleXtalMinEnergy = cms.double( -0.15 ),
    seleXtalMinEnergyEndCap = cms.double( -0.75 ),
    selePtGamma = cms.double(0.8 ),
    selePtPi0 = cms.double( 1.6 ),
    seleMinvMaxPi0 = cms.double( 0.26 ),
    seleMinvMinPi0 = cms.double( 0.04 ),
    seleS4S9Gamma = cms.double( 0.83 ),
    selePi0Iso = cms.double( 0.5 ),
    ptMinForIsolation = cms.double( 0.5 ),
    selePi0BeltDR = cms.double( 0.2 ),
    selePi0BeltDeta = cms.double( 0.05 ),

    selePtGammaEndCap = cms.double( 0.5 ),
    selePtPi0EndCap = cms.double( 2.0 ),
    seleS4S9GammaEndCap = cms.double( 0.9 ),
    seleMinvMaxPi0EndCap = cms.double( 0.3 ),
    seleMinvMinPi0EndCap = cms.double( 0.05 ),
    ptMinForIsolationEndCap = cms.double( 0.5 ),
    selePi0IsoEndCap = cms.double( 0.5 ),
    selePi0BeltDREndCap  = cms.double( 0.2 ),
    selePi0BeltDetaEndCap  = cms.double( 0.05 ),

    selePtGammaEta = cms.double(0.8),
    selePtEta = cms.double(3.0),
    seleS4S9GammaEta  = cms.double(0.87),
    seleS9S25GammaEta  = cms.double(0.8),
    seleMinvMaxEta = cms.double(0.9),
    seleMinvMinEta = cms.double(0.2),
    ptMinForIsolationEta = cms.double(0.5),
    seleEtaIso = cms.double(0.5),
    seleEtaBeltDR = cms.double(0.3),
    seleEtaBeltDeta = cms.double(0.1),
    massLowPi0Cand = cms.double(0.084),
    massHighPi0Cand = cms.double(0.156),

    selePtGammaEtaEndCap = cms.double(1.0),
    selePtEtaEndCap = cms.double(3.0),
    seleS4S9GammaEtaEndCap  = cms.double(0.9),
    seleS9S25GammaEtaEndCap  = cms.double(0.85),
    seleMinvMaxEtaEndCap = cms.double(0.9),
    seleMinvMinEtaEndCap = cms.double(0.2),
    ptMinForIsolationEtaEndCap = cms.double(0.5),
    seleEtaIsoEndCap = cms.double(0.5),
    seleEtaBeltDREndCap = cms.double(0.3),
    seleEtaBeltDetaEndCap = cms.double(0.1),
                          
    ParameterLogWeighted = cms.bool( True ),
    ParameterX0 = cms.double( 0.89 ),
    ParameterT0_barl = cms.double( 5.7 ),
    ParameterT0_endc = cms.double( 3.1 ),
    ParameterT0_endcPresh = cms.double( 1.2 ),
    ParameterW0 = cms.double( 4.2 )

)

                          
