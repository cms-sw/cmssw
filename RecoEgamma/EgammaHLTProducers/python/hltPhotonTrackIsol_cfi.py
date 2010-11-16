import FWCore.ParameterSet.Config as cms

hltPhotonTrackIsol = cms.EDProducer("EgammaHLTPhotonTrackIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag("hltEgammaHcalIsolFilter"),
    trackProducer = cms.InputTag("ctfWithMaterialTracks"),
    countTracks = cms.bool(False),
    egTrkIsoPtMin = cms.double(1.5),
    egTrkIsoConeSize = cms.double(0.3),
    egTrkIsoZSpan = cms.double(999999.0),
    egTrkIsoRSpan = cms.double(999999.0),
    egTrkIsoVetoConeSize = cms.double(0.0)
)

