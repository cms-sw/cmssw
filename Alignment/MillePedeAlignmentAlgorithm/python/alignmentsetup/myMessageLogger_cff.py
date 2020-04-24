import FWCore.ParameterSet.Config as cms


#####################################################################
## MessageLogger for convenient output
######################################################################
MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('alignment'), ##, 'cout')
    categories = cms.untracked.vstring(
        'Alignment', 
        'LogicError', 
        'FwkReport', 
        'TrackProducer'),
    alignment = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('DEBUG'),
        LogicError = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        Alignment = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    destinations = cms.untracked.vstring('alignment') ## (, 'cout')
)
