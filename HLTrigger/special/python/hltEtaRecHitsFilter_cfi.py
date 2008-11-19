import FWCore.ParameterSet.Config as cms



hltAlCaEtaRegRecHits = cms.EDFilter( "HLTEtaRecHitsFilter",
    barrelHits = cms.InputTag( 'hltEcalRegionalEtaRecHit','EcalRecHitsEB' ),
                                     
    etaBarrelHitCollection = cms.string( "etaEcalRecHitsEB" ),

    clusSeedThr = cms.double( 0.5 ),
    clusEtaSize = cms.int32( 3 ),
    clusPhiSize = cms.int32( 3 ),
    seleXtalMinEnergy = cms.double( 0.0 ),

    ptMinEMObj = cms.double( 2.0 ),
    EMregionEtaMargin = cms.double( 0.25 ),
    EMregionPhiMargin = cms.double( 0.4 ),

    RegionalMatch = cms.untracked.bool( True ),
    Jets = cms.untracked.bool( False ),
  
    selePtGammaEta = cms.double(1.2),
    selePtEta = cms.double(4.0),                                     
    seleS4S9GammaEta  = cms.double(0.9),                                 
    seleMinvMaxEta = cms.double(0.7),
    seleMinvMinEta = cms.double(0.35),                                   
    ptMinForIsolationEta = cms.double(1.0),
    seleIsoEta = cms.double(0.2),
    seleEtaBeltDR = cms.double(0.3),
    seleEtaBeltDeta = cms.double(0.1),
    storeIsoClusRecHitEta = cms.bool(True),
    removePi0CandidatesForEta = cms.bool(True),
    massLowPi0Cand = cms.double(0.114),
    massHighPi0Cand = cms.double(0.154),                                       

    ParameterLogWeighted = cms.bool( True ),
    ParameterX0 = cms.double( 0.89 ),
    ParameterT0_barl = cms.double( 7.4 ),
    ParameterT0_endc = cms.double( 3.1 ),
    ParameterT0_endcPresh = cms.double( 1.2 ),
    ParameterW0 = cms.double( 4.2 ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    l1SeedFilterTag = cms.InputTag( "hltL1sAlCaEcalEta" ),
    debugLevel = cms.int32( 0 )
  
)
