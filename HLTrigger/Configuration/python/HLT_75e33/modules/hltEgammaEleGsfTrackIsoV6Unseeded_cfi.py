import FWCore.ParameterSet.Config as cms

hltEgammaEleGsfTrackIsoV6Unseeded = cms.EDProducer("EgammaHLTElectronTrackIsolationProducers",
    beamSpotProducer = cms.InputTag("hltOnlineBeamSpot"),
    egTrkIsoConeSize = cms.double(0.3),
    egTrkIsoPtMin = cms.double(1.0),
    egTrkIsoRSpan = cms.double(999999.0),
    egTrkIsoStripBarrel = cms.double(0.01),
    egTrkIsoStripEndcap = cms.double(0.01),
    egTrkIsoVetoConeSizeBarrel = cms.double(0.01),
    egTrkIsoVetoConeSizeEndcap = cms.double(0.01),
    egTrkIsoZSpan = cms.double(0.15),
    electronProducer = cms.InputTag("hltEgammaGsfElectronsUnseeded"),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesUnseeded"),
    trackProducer = cms.InputTag("generalTracks"),
    useGsfTrack = cms.bool(True),
    useSCRefs = cms.bool(True)
)
