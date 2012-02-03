import FWCore.ParameterSet.Config as cms

hltEgammaGsfTrackVar = cms.EDProducer( "EgammaHLTGsfTrackVarProducer",
    inputCollection = cms.InputTag( "hltL1GsfElectrons" ),
    recoEcalCandidateProducer = cms.InputTag("hltL1RecoEcalCandidate"),
    beamSpotProducer = cms.InputTag( "hltOnlineBeamSpot" ),
    upperTrackNrToRemoveCut = cms.int32(9999),
    lowerTrackNrToRemoveCut = cms.int32(-1),
)
