import FWCore.ParameterSet.Config as cms

hltPhase2MuonIOTrackCandidatesMkFit = cms.EDProducer("MkFitProducer",
        backwardFitInCMSSW = cms.bool(False),
        buildingRoutine = cms.string('cloneEngine'),
        clustersToSkip = cms.InputTag(""),
        config = cms.ESInputTag("","hltInitialStepTrackCandidatesMkFitConfig"),
        eventOfHits = cms.InputTag("hltMkFitEventOfHits"),
        limitConcurrency = cms.untracked.bool(False),
        mightGet = cms.optional.untracked.vstring,
        minGoodStripCharge = cms.PSet(),
        mkFitSilent = cms.untracked.bool(True),
        pixelHits = cms.InputTag("hltMkFitSiPixelHits"),
        removeDuplicates = cms.bool(True),
        seedCleaning = cms.bool(True),
        seeds = cms.InputTag("hltPhase2MuonIOTrackMkFitSeeds"),
        stripHits = cms.InputTag("hltMkFitSiPhase2Hits")
)
