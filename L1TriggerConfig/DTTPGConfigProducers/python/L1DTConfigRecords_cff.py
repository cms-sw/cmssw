import FWCore.ParameterSet.Config as cms

rcdsrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('DTConfigManagerRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


# foo bar baz
# CKFMc5KLRVDf7
# DATE697tQwjlf
