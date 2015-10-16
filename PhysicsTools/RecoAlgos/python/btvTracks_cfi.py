import FWCore.ParameterSet.Config as cms

import CommonTools.RecoAlgos.recoTrackSelectorPSet_cfi as _recoTrackSelectorPSet_cfi

_content = _recoTrackSelectorPSet_cfi.recoTrackSelectorPSet.clone(
    maxChi2 = 5.0,
    tip = 0.2,
    minRapidity = -9.0,
    lip = 17.0,
    ptMin = 1.0,
    maxRapidity = 9.0,
    quality = [],
    minLayer = 0,
    minHit = 8,
    minPixelHit = 2,
    usePV = True,
)

btvTracks = cms.EDProducer("RecoTrackSelector",
    _content,
    copyExtras = cms.untracked.bool(True), ## copies also extras and rechits on RECO
    copyTrajectories = cms.untracked.bool(False) # don't set this to true on AOD!
)

btvTrackRefs = cms.EDProducer("RecoTrackViewRefSelector",
    _content
)


