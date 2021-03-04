# This cfi sets a field map configured based on the run and GT.
#
# PLEASE DO NOT USE THIS cfi DIRECTLY
# Always use the standard sequence Configuration.StandardSequences.MagneticField_cff

import FWCore.ParameterSet.Config as cms

VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducerFromDB",
    label = cms.untracked.string(''),
    debugBuilder = cms.untracked.bool(False),
    valueOverride = cms.int32(-1), # Force value of current (in A); take the value from DB if < 0.
)

_VolumeBasedMagneticFieldESProducer_dd4hep = cms.ESProducer("DD4hep_VolumeBasedMagneticFieldESProducerFromDB",
    label = cms.untracked.string(''),
    debugBuilder = cms.untracked.bool(False),
    valueOverride = cms.int32(-1), # Force value of current (in A); take the value from DB if < 0.
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toReplaceWith(VolumeBasedMagneticFieldESProducer, _VolumeBasedMagneticFieldESProducer_dd4hep)
