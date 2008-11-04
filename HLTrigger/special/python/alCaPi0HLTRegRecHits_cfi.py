import FWCore.ParameterSet.Config as cms

# AlCaPi0RecHits HLT filter
alCaPi0RegRecHits = cms.EDFilter("HLTPi0RecHitsFilter",
    barrelHits = cms.InputTag( 'hltEcalRegionalPi0RecHit','EcalRecHitsEB' ),
    endcapHits = cms.InputTag( 'hltEcalRegionalPi0RecHit','EcalRecHitsEE' ),
                                     
    pi0BarrelHitCollection = cms.string( "pi0EcalRecHitsEB" ),
    pi0EndcapHitCollection = cms.string( "pi0EcalRecHitsEE" ),                                

    clusSeedThr = cms.double( 0.5 ),
    clusSeedThrEndCap = cms.double( 1.0 ),                            
    clusEtaSize = cms.int32( 3 ),
    clusPhiSize = cms.int32( 3 ),
    selePtGammaOne = cms.double( 0.9 ),
    selePtGammaTwo = cms.double( 0.9 ),
    selePtPi0 = cms.double( 2. ),
    seleMinvMaxPi0 = cms.double( 0.22 ),
    seleMinvMinPi0 = cms.double( 0.06 ),
    seleXtalMinEnergy = cms.double( 0.0 ),
    seleNRHMax = cms.int32( 1000 ),
    seleS4S9GammaOne = cms.double( 0.8 ),
    seleS4S9GammaTwo = cms.double( 0.8 ),
    selePi0Iso = cms.double( 1.0 ),
    selePi0BeltDR = cms.double( 0.2 ),
    selePi0BeltDeta = cms.double( 0.05 ),
    ptMinForIsolation = cms.double( 0.9 ),
    storeIsoClusRecHit = cms.bool( True ),
                                     
    ptMinEMObj = cms.double( 2.0 ),
    EMregionEtaMargin = cms.double( 0.25 ),
    EMregionPhiMargin = cms.double( 0.4 ),

    RegionalMatch = cms.untracked.bool( True ),
    Jets = cms.untracked.bool( False ),
                                     
    selePtGammaEndCap = cms.double( 0.8 ),
    selePtPi0EndCap = cms.double( 2.0 ),
    seleS4S9GammaEndCap = cms.double( 0.85 ),                                 
    seleMinvMaxPi0EndCap = cms.double( 0.3 ),
    seleMinvMinPi0EndCap = cms.double( 0.05 ),                                 
    ptMinForIsolationEndCap = cms.double( 0.5 ),
    selePi0IsoEndCap = cms.double( 1.0 ),                                    

    doSelForEtaBarrel = cms.untracked.bool( True ),
    selePtGammaEta = cms.double(1.),
    selePtEta = cms.double(3.5),                                     
    seleS4S9GammaEta  = cms.double(0.88),                                 
    seleMinvMaxEta = cms.double(0.7),
    seleMinvMinEta = cms.double(0.4),                                   
    ptMinForIsolationEta = cms.double(1.0),
    seleIsoEta = cms.double(0.5),
    seleEtaBeltDR = cms.double(0.3),
    seleEtaBeltDeta = cms.double(0.1),
    storeIsoClusRecHitEta = cms.bool( True ),                                   

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


