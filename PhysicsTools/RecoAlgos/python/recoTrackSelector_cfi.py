import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.recoTrackSelectorPSet_cfi import recoTrackSelectorPSet as _recoTrackSelectorPSet

recoTrackSelector = cms.EDProducer("RecoTrackSelector",
    _recoTrackSelectorPSet,
    copyExtras = cms.untracked.bool(True), ## copies also extras and rechits on RECO
    copyTrajectories = cms.untracked.bool(False) # don't set this to true on AOD!
)



