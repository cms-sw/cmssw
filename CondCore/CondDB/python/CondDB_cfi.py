import FWCore.ParameterSet.Config as cms

CondDB = cms.PSet(
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        messageLevel = cms.untracked.int32(0),
    ),
    connect = cms.string('protocol://db/schema') ##db/schema"
)

