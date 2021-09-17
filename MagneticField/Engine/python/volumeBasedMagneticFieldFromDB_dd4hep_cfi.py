# This cfi sets a field map configured based on the run and GT.
#
# PLEASE DO NOT USE THIS cfi DIRECTLY
# Always use the standard sequence Configuration.StandardSequences.MagneticField_cff

import FWCore.ParameterSet.Config as cms

VolumeBasedMagneticFieldESProducer = cms.ESProducer("DD4hep_VolumeBasedMagneticFieldESProducerFromDB",
    label = cms.untracked.string(''),
    debugBuilder = cms.untracked.bool(False),
    valueOverride = cms.int32(-1), # Force value of current (in A); take the value from DB if < 0.
)

