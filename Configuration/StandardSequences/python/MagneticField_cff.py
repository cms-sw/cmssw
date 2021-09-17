# This is the default configuration for the magnetic field in CMSSW.
# Loads the field map corresponding to the current stored in the runInfo,
# with the geometry and configuration specified in the GT.

import FWCore.ParameterSet.Config as cms
from MagneticField.Engine.volumeBasedMagneticFieldFromDB_cfi import *

from MagneticField.Engine.volumeBasedMagneticFieldFromDB_dd4hep_cfi import VolumeBasedMagneticFieldESProducer as _VolumeBasedMagneticFieldESProducer_dd4hep

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toReplaceWith(VolumeBasedMagneticFieldESProducer, _VolumeBasedMagneticFieldESProducer_dd4hep)

# Parabolic parametrized magnetic field used for track building (scaled to nominal map closest to current from runInfo)
from MagneticField.ParametrizedEngine.autoParabolicParametrizedField_cfi import ParametrizedMagneticFieldProducer as ParabolicParametrizedMagneticFieldProducer
ParabolicParametrizedMagneticFieldProducer.label = "ParabolicMf"


