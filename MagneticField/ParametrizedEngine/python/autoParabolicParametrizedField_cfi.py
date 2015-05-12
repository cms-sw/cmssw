import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use the parabolic magnetic
# field engine (tracker region only) scaled to the nominal value closest to the current read from RunInfo.

idealMagneticFieldRecordSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('IdealMagneticFieldRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

ParametrizedMagneticFieldProducer = cms.ESProducer("AutoParametrizedMagneticFieldProducer",
    version = cms.string('Parabolic'),
    label = cms.untracked.string(''),
    valueOverride = cms.int32(-1)
)

