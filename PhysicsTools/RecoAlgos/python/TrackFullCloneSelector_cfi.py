import FWCore.ParameterSet.Config as cms

trackFullCloneSelector = cms.EDFilter("TrackFullCloneSelector",
    copyExtras = cms.untracked.bool(False), ## copies also extras and rechits on RECO

    src = cms.InputTag("ctfWithMaterialTracks"),
    cut = cms.string('(numberOfValidHits >= 8) & (normalizedChi2 < 5)'),
    # don't set this to true on AOD!
    copyTrajectories = cms.untracked.bool(False)
)


