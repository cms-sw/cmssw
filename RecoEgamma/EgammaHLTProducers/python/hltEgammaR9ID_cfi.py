import FWCore.ParameterSet.Config as cms

hltEgammaR9IDShape = cms.EDProducer( "EgammaHLTR9IDProducer",
   recoEcalCandidateProducer = cms.InputTag( "hltRecoEcalCandidate" ),
   ecalRechitEB = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
   ecalRechitEE = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
)
