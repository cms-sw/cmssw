import FWCore.ParameterSet.Config as cms

#
# The following settings load all correction classes
# stored in the database
#
OOTPileupCorrectionESProducer = cms.ESProducer(
    'OOTPileupCorrectionESProducer',
    name = cms.string(".*"),
    nameIsRegex = cms.bool(True),
    category = cms.string(".*"),
    categoryIsRegex = cms.bool(True)
)
