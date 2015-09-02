import FWCore.ParameterSet.Config as cms

CondDB = cms.PSet(
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        authenticationSystem = cms.untracked.int32(0),
        messageLevel = cms.untracked.int32(0),
    ),
    connect = cms.string(''), 
    dbFormat = cms.untracked.int32(0)
)

