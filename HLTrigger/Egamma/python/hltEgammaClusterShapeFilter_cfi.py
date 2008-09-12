import FWCore.ParameterSet.Config as cms

hltEgammaClusterShapeFilter = cms.EDFilter( "HLTEgammaClusterShapeFilter",
    candTag = cms.InputTag( "hltEgammaEtFilter" ),
    ecalRechitEB = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),                                     
    ecalRechitEE = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),                                     
    ncandcut = cms.int32( 1 ),
    BarrelThreshold = cms.double( 0.014 ),
    EndcapThreshold = cms.double( 0.027 ),                                
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
