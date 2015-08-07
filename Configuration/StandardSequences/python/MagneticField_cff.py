# This is the default configuration for the magnetic field in CMSSW.
# Loads the field map corresponding to the current stored in the runInfo,
# with the geometry and configuration specified in the GT.

import FWCore.ParameterSet.Config as cms
from MagneticField.Engine.volumeBasedMagneticFieldFromDB_cfi import *

# Parabolic parametrized magnetic field used for track building (scaled to nominal map closest to current from runInfo)
from MagneticField.ParametrizedEngine.autoParabolicParametrizedField_cfi import ParametrizedMagneticFieldProducer as ParabolicParametrizedMagneticFieldProducer
ParabolicParametrizedMagneticFieldProducer.label = "ParabolicMf"


