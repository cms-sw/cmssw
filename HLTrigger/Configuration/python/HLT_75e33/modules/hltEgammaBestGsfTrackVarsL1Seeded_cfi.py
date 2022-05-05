import FWCore.ParameterSet.Config as cms

hltEgammaBestGsfTrackVarsL1Seeded = cms.EDProducer("EgammaHLTGsfTrackVarProducer",
    beamSpotProducer = cms.InputTag("hltOnlineBeamSpot"),
    inputCollection = cms.InputTag("hltEgammaGsfElectronsL1Seeded"),
    lowerTrackNrToRemoveCut = cms.int32(-1),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    upperTrackNrToRemoveCut = cms.int32(9999),
    useDefaultValuesForBarrel = cms.bool(False),
    useDefaultValuesForEndcap = cms.bool(False)
)
