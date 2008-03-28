import FWCore.ParameterSet.Config as cms

#
# producer for hltPhotonTrackIsol
#
hltPhotonTrackIsol = cms.EDFilter("EgammaHLTPhotonTrackIsolationProducersRegional",
    egTrkIsoVetoConeSize = cms.double(0.0),
    trackProducer = cms.InputTag("ctfWithMaterialTracks"),
    egTrkIsoConeSize = cms.double(0.3),
    egTrkIsoRSpan = cms.double(999999.0),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaHcalIsolFilter"),
    #InputTag trackProducer       = hltSingleCtfWithMaterialTracks
    egTrkIsoPtMin = cms.double(1.5),
    egTrkIsoZSpan = cms.double(999999.0)
)


