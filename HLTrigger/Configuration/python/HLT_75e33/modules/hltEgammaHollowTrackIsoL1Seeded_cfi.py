import FWCore.ParameterSet.Config as cms

hltEgammaHollowTrackIsoL1Seeded = cms.EDProducer("EgammaHLTPhotonTrackIsolationProducersRegional",
    countTracks = cms.bool(False),
    egTrkIsoConeSize = cms.double(0.29),
    egTrkIsoPtMin = cms.double(1.0),
    egTrkIsoRSpan = cms.double(999999.0),
    egTrkIsoStripBarrel = cms.double(0.03),
    egTrkIsoStripEndcap = cms.double(0.03),
    egTrkIsoVetoConeSize = cms.double(0.06),
    egTrkIsoZSpan = cms.double(999999.0),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    trackProducer = cms.InputTag("generalTracks")
)
