import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.recoTrackSelectorPSet_cfi import recoTrackSelectorPSet as _recoTrackSelectorPSet

recoTrackViewRefSelector = cms.EDProducer("RecoTrackViewRefSelector",
    _recoTrackSelectorPSet
)
# foo bar baz
# FajDQPgrprcky
# 99wVkvBj0XQDl
