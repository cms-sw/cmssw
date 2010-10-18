import FWCore.ParameterSet.Config as cms

hltAlCaEcalPi0Filterv1 = cms.EDFilter( "HLTEcalResonanceFilter",
    barrelHits = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEB' ),
    endcapHits = cms.InputTag( 'hltEcalRegionalPi0EtaRecHit','EcalRecHitsEE' ),                                  
    barrelClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersBarrel'),
    endcapClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersEndcap'),
    useRecoFlag = cms.bool( False ),
    flagLevelRecHitsToUse = cms.int32( 1 ),
    useDBStatus = cms.bool( True ),
    statusLevelRecHitsToUse = cms.int32( 1 ),                                        

                                       
    doSelBarrel = cms.bool( True ),
    barrelSelection = cms.PSet(                                   
       selePtGamma = cms.double( 1 ),
       selePtPair = cms.double( 2 ),
       seleMinvMaxBarrel = cms.double( 0.22 ),
       seleMinvMinBarrel = cms.double( 0.06 ),
       removePi0CandidatesForEta = cms.bool ( False),
       massLowPi0Cand = cms.double( 0.104 ),
       massHighPi0Cand = cms.double( 0.163 ),                                      
       seleS4S9Gamma = cms.double( 0.83 ),
       seleS9S25Gamma = cms.double( 0 ),
       ptMinForIsolation = cms.double( 1.0 ),
       seleIso = cms.double( 0.5 ),
       seleBeltDR = cms.double( 0.2 ),
       seleBeltDeta = cms.double( 0.05 ),                                         
       store5x5RecHitEB = cms.bool( False ),
       barrelHitCollection = cms.string( "pi0EcalRecHitsEB" )
    ),
   
    doSelEndcap = cms.bool( True ),
    endcapSelection = cms.PSet(
       seleMinvMaxEndCap = cms.double( 0.3 ),
       seleMinvMinEndCap = cms.double( 0.05 ),                                         
       region1_EndCap = cms.double( 2.0 ),
       selePtGammaEndCap_region1 = cms.double( 0.8 ),
       selePtPairEndCap_region1 = cms.double( 3.0 ),
       region2_EndCap = cms.double( 2.5 ),
       selePtGammaEndCap_region2 = cms.double( 0.5 ),
       selePtPairEndCap_region2 = cms.double( 2.0 ),
       selePtGammaEndCap_region3 = cms.double( 0.3 ),
       selePtPairEndCap_region3 = cms.double( 1.2 ),
       selePtPairMaxEndCap_region3 = cms.double( 2.5 ),
       seleS4S9GammaEndCap = cms.double( 0.9 ),
       seleS9S25GammaEndCap = cms.double( 0 ),
       ptMinForIsolationEndCap = cms.double( 0.5 ),
       seleIsoEndCap = cms.double( 0.5 ),
       seleBeltDREndCap = cms.double( 0.2 ),
       seleBeltDetaEndCap = cms.double( 0.05 ),                                         
       store5x5RecHitEE = cms.bool( False ),
       endcapHitCollection = cms.string( "pi0EcalRecHitsEE" )
    ),

    preshRecHitProducer = cms.InputTag( 'hltESRegionalPi0EtaRecHit','EcalRecHitsES' ),
    storeRecHitES = cms.bool( True ),
    preshowerSelection = cms.PSet(
       ESCollection = cms.string( "pi0EcalRecHitsES" ),                                         
       preshNclust = cms.int32( 4 ),
       preshClusterEnergyCut = cms.double( 0.0 ),
       preshStripEnergyCut = cms.double( 0.0 ),
       preshSeededNstrip = cms.int32( 15 ),
       preshCalibPlaneX = cms.double( 1.0 ),
       preshCalibPlaneY = cms.double( 0.7 ),
       preshCalibGamma = cms.double( 0.024 ),
       preshCalibMIP = cms.double( 9.0E-5 ),
       debugLevelES = cms.string( " " )

   ),
                                       
    debugLevel = cms.int32( 0 )
                                 
   
)
