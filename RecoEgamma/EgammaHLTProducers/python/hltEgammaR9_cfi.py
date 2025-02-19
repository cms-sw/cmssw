import FWCore.ParameterSet.Config as cms

hltEgammaR9Shape = cms.EDProducer( "EgammaHLTR9Producer",
   recoEcalCandidateProducer = cms.InputTag( "hltRecoEcalCandidate" ),
   ecalRechitEB = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
   ecalRechitEE = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
   useSwissCross = cms.bool( False )                                
)
