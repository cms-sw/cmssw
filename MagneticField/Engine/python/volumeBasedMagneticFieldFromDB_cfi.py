import FWCore.ParameterSet.Config as cms

VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducerFromDB",
    label = cms.untracked.string(''),
    debugBuilder = cms.untracked.bool(False),
    valueOverride = cms.int32(-1), # Force value of current (in A); take the value from DB if < 0.
)

