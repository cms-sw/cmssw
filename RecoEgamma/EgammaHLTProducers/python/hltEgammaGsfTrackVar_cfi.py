import FWCore.ParameterSet.Config as cms

hltEgammaGsfTrackVar = cms.EDProducer( "EgammaHLTGsfTrackVarProducer",
    inputCollection = cms.InputTag( "hltL1GsfElectrons" ),
    recoEcalCandidateProducer = cms.InputTag("hltL1RecoEcalCandidate"),
    beamSpotProducer = cms.InputTag( "hltOnlineBeamSpot" )
)
