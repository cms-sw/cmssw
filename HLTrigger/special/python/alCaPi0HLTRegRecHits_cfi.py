import FWCore.ParameterSet.Config as cms

# AlCaPi0RecHits HLT filter
hltAlCaPi0RegRecHits = cms.EDFilter( "HLTPi0RecHitsFilter",

    barrelHits = cms.InputTag( 'hltEcalRegionalPi0RecHit','EcalRecHitsEB' ),
    endcapHits = cms.InputTag( 'hltEcalRegionalPi0RecHit','EcalRecHitsEE' ),
                                     

    clusSeedThr = cms.double( 0.5 ),
    clusSeedThrEndCap = cms.double( 1.0 ),                            
    clusEtaSize = cms.int32( 3 ),
    clusPhiSize = cms.int32( 3 ),
    seleXtalMinEnergy = cms.double( -0.15 ),
    seleXtalMinEnergyEndCap = cms.double( -0.75 ),
    doSelForPi0Barrel = cms.bool( True ),
    selePtGamma = cms.double(1 ),
    selePtPi0 = cms.double( 2. ),
    seleMinvMaxPi0 = cms.double( 0.22 ),
    seleMinvMinPi0 = cms.double( 0.06 ),
    seleS4S9Gamma = cms.double( 0.83 ),
    selePi0Iso = cms.double( 0.5 ),
    ptMinForIsolation = cms.double( 1 ),
    selePi0BeltDR = cms.double( 0.2 ),
    selePi0BeltDeta = cms.double( 0.05 ),                                 
    storeIsoClusRecHitPi0EB = cms.bool( True ),
    pi0BarrelHitCollection = cms.string( "pi0EcalRecHitsEB" ),                           

    
    doSelForPi0Endcap = cms.bool( True ),
    selePtGammaEndCap = cms.double( 0.8 ),
    selePtPi0EndCap = cms.double( 3.0 ),
    seleS4S9GammaEndCap = cms.double( 0.9 ),                                 
    seleMinvMaxPi0EndCap = cms.double( 0.3 ),
    seleMinvMinPi0EndCap = cms.double( 0.05 ),                                 
    ptMinForIsolationEndCap = cms.double( 0.5 ),
    selePi0IsoEndCap = cms.double( 0.5 ),
    selePi0BeltDREndCap  = cms.double( 0.2 ),
    selePi0BeltDetaEndCap  = cms.double( 0.05 ),                                   
    storeIsoClusRecHitPi0EE = cms.bool (True),
    pi0EndcapHitCollection = cms.string( "pi0EcalRecHitsEE" ),
                           
                                 
    doSelForEtaBarrel = cms.bool( True ),
    selePtGammaEta = cms.double(1.2),
    selePtEta = cms.double(4.0),                                     
    seleS4S9GammaEta  = cms.double(0.9),
    seleS9S25GammaEta  = cms.double(0.8),                                 
    seleMinvMaxEta = cms.double(0.8),
    seleMinvMinEta = cms.double(0.3),                                   
    ptMinForIsolationEta = cms.double(1.0),
    seleEtaIso = cms.double(0.5),
    seleEtaBeltDR = cms.double(0.3),
    seleEtaBeltDeta = cms.double(0.1),
    storeIsoClusRecHitEtaEB = cms.bool(True),                                    
    removePi0CandidatesForEta = cms.bool(True),
    massLowPi0Cand = cms.double(0.104),
    massHighPi0Cand = cms.double(0.163),
    store5x5RecHitEtaEB = cms.bool(True),                               
    etaBarrelHitCollection = cms.string( "etaEcalRecHitsEB" ),
                                 
                                         
    doSelForEtaEndcap = cms.bool( True ),
    selePtGammaEtaEndCap = cms.double(1.5),
    selePtEtaEndCap = cms.double(5),                                     
    seleS4S9GammaEtaEndCap  = cms.double(0.9),                                 
    seleS9S25GammaEtaEndCap  = cms.double(0.85),    
    seleMinvMaxEtaEndCap = cms.double(0.8),
    seleMinvMinEtaEndCap = cms.double(0.3),                                   
    ptMinForIsolationEtaEndCap = cms.double(0.5),
    seleEtaIsoEndCap = cms.double(0.5),
    seleEtaBeltDREndCap = cms.double(0.3),
    seleEtaBeltDetaEndCap = cms.double(0.1),
    storeIsoClusRecHitEtaEE = cms.bool(True),                                    
    store5x5RecHitEtaEE = cms.bool(True),                               
    etaEndcapHitCollection = cms.string( "etaEcalRecHitsEE" ),

    seleNRHMax = cms.int32( 1000 ),
    ptMinEMObj = cms.double( 2.0 ),
    EMregionEtaMargin = cms.double( 0.25 ),
    EMregionPhiMargin = cms.double( 0.4 ),
    RegionalMatch = cms.untracked.bool( True ),
    Jets = cms.untracked.bool( False ),

    ParameterLogWeighted = cms.bool( True ),
    ParameterX0 = cms.double( 0.89 ),
    ParameterT0_barl = cms.double( 5.7 ),
    ParameterT0_endc = cms.double( 3.1 ),
    ParameterT0_endcPresh = cms.double( 1.2 ),
    ParameterW0 = cms.double( 4.2 ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    l1SeedFilterTag = cms.InputTag( "hltL1sAlCaEcalPi0" ),

    debugLevel = cms.int32( 0 )

                                     
)
