import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.recoTrackSelectorPSet_cfi import recoTrackSelectorPSet as _recoTrackSelectorPSet

recoTrackRefSelector = cms.EDProducer("RecoTrackRefSelector",
    _recoTrackSelectorPSet
)



