### THIS CFF IS DEPRECATED!!!! ###
# please use MagneticField_cff.py instead

# Run 2 field map configuration for 3.8T, version 120812.
import FWCore.ParameterSet.Config as cms
from MagneticField.Engine.volumeBasedMagneticField_120812_largeYE4_cfi import *

# Parabolic parametrized magnetic field used for track building
from MagneticField.ParametrizedEngine.ParabolicParametrizedField_cfi import ParametrizedMagneticFieldProducer as ParabolicParametrizedMagneticFieldProducer
ParabolicParametrizedMagneticFieldProducer.label = "ParabolicMf"

