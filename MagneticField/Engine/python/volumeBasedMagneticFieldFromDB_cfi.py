import FWCore.ParameterSet.Config as cms

#FIXME
ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('OAE_1103l_071212'),
    parameters = cms.PSet(
        BValue = cms.string('3_8T')
    ),
    label = cms.untracked.string('parametrizedField')
)

VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducerFromDB",
    label = cms.untracked.string(''),
    debugBuilder = cms.untracked.bool(False),
    valueOverride = cms.int32(-1), # Force value of current (in A); take the value from DB if < 0.
)

