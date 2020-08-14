import FWCore.ParameterSet.Config as cms

from MagneticField.Engine.volumeBasedMagneticFieldFromDB_cfi import (
    VolumeBasedMagneticFieldESProducer as _VolumeBasedMagneticFieldESProducer,
)

hltPhase2VolumeBasedMagneticFieldESProducer = (
    _VolumeBasedMagneticFieldESProducer.clone()
)
