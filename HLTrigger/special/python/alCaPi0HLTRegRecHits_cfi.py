import FWCore.ParameterSet.Config as cms

# AlCaPi0RecHits HLT filter
alCaPi0RegRecHits = cms.EDFilter("HLTPi0RecHitsFilter",
    barrelHits = cms.InputTag( 'hltEcalRegionalEgammaRecHitLowPT','EcalRecHitsEB' ),
    endcapHits = cms.InputTag( 'hltEcalRegionalEgammaRecHitLowPT','EcalRecHitsEE' ),
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
    selePi0IsoEndCap = cms.double( 1.0 ),                                 
    selePi0BeltDR = cms.double( 0.2 ),
    selePi0BeltDeta = cms.double( 0.05 ),
    ptMinForIsolation = cms.double( 0.9 ),
                                 
    Jets = cms.untracked.bool( False ),
    JETSdoCentral =  cms.untracked.bool( True ),
    JETSdoForward =  cms.untracked.bool( True ),
    JETSdoTau =  cms.untracked.bool( True ),         
    Ptmin_jets = cms.untracked.double (20.0),
    CentralSource = cms.untracked.InputTag( 'hltL1extraParticles','Central' ),
    ForwardSource = cms.untracked.InputTag( 'hltL1extraParticles','Forward' ),
    TauSource = cms.untracked.InputTag( 'hltL1extraParticles','Tau' ),                                    

    selePtGammaEndCap = cms.double( 0.8 ),
    selePtPi0EndCap = cms.double( 2.0 ),
    seleS4S9GammaEndCap = cms.double( 0.85 ),                                 
    seleMinvMaxPi0EndCap = cms.double( 0.3 ),
    seleMinvMinPi0EndCap = cms.double( 0.05 ),                                 
    ptMinForIsolationEndCap = cms.double( 0.5 ),
                                 
    ParameterLogWeighted = cms.bool( True ),
    ParameterX0 = cms.double( 0.89 ),
    ParameterT0_barl = cms.double( 5.7 ),
    ParameterT0_endc = cms.double( 3.1 ),
    ParameterT0_endcPresh = cms.double( 1.2 ),
    ParameterW0 = cms.double( 4.2 ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    l1SeedFilterTag = cms.InputTag( "hltL1sAlCaEcalPi0" ),
    debugLevel = cms.int32( 0 ),
    storeIsoClusRecHit = cms.bool( True ),

    ptMinEMObj = cms.double( 2.0 ),
    EMregionEtaMargin = cms.double( 0.25 ),
    EMregionPhiMargin = cms.double( 0.4 ) 
)


