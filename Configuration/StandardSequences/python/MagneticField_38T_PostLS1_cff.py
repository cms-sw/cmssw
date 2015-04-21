# Run 2 default field map configuration. This cff is DEPRECATED, please use MagneticField_AutoFromDBCurrent_cff.py

import FWCore.ParameterSet.Config as cms
from MagneticField.Engine.volumeBasedMagneticField_120812_largeYE4_cfi import *

# Parabolic parametrized magnetic field used for track building
from MagneticField.ParametrizedEngine.ParabolicParametrizedField_cfi import ParametrizedMagneticFieldProducer as ParabolicParametrizedMagneticFieldProducer
ParabolicParametrizedMagneticFieldProducer.label = "ParabolicMf"

