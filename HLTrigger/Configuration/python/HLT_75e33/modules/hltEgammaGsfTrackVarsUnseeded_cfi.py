import FWCore.ParameterSet.Config as cms

hltEgammaGsfTrackVarsUnseeded = cms.EDProducer("EgammaHLTGsfTrackVarProducer",
    beamSpotProducer = cms.InputTag("hltOnlineBeamSpot"),
    inputCollection = cms.InputTag("hltEgammaGsfTracksUnseeded"),
    lowerTrackNrToRemoveCut = cms.int32(-1),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesUnseeded"),
    upperTrackNrToRemoveCut = cms.int32(9999),
    useDefaultValuesForBarrel = cms.bool(False),
    useDefaultValuesForEndcap = cms.bool(False)
)
