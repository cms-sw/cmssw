import FWCore.ParameterSet.Config as cms

highPtTrackRefs = cms.EDFilter("TrackRefSelector",
    src = cms.InputTag("ctfWithMaterialTracks"),
    cut = cms.string('pt > 20')
)


# foo bar baz
# zIEQmR6Pr7l6D
# dPShHCrWkhKwZ
