import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        TrackTransformer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        RecoMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),      
        TrackFitters = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('DEBUG')
    ),
    categories = cms.untracked.vstring('RecoMuon', 
        'TrackTransformer', 
        'TrackFitters'),
    destinations = cms.untracked.vstring('cout')
)



