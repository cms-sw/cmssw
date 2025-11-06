import FWCore.ParameterSet.Config as cms

hltInitialStepTrackCandidatesMkFitFit = cms.EDProducer("MkFitFitProducer",
    config = cms.ESInputTag("","hltInitialStepTrackCandidatesMkFitConfig"),
    eventOfHits = cms.InputTag("hltMkFitEventOfHits"),
    limitConcurrency = cms.untracked.bool(False),
    mightGet = cms.optional.untracked.vstring,
    mkFitPixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    mkFitSilent = cms.untracked.bool(True),
    pixelCPE = cms.string('PixelCPEGeneric'),
    pixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    stripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    tracks = cms.InputTag("hltInitialStepTrackCandidatesMkFit"),
    candCutSel = cms.bool(True),
    candMinNHitsCut = cms.int32(3),
    candMinPtCut = cms.double(0.8)
)
