import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use the VolumeBased magnetic
# field engine.
#Default is version 85l
from MagneticField.Engine.volumeBasedMagneticField_120812_largeYE4_cfi import *

# Parabolic parametrized magnetic field used for track building
from MagneticField.ParametrizedEngine.ParabolicParametrizedField_cfi import ParametrizedMagneticFieldProducer as ParabolicParametrizedMagneticFieldProducer
ParabolicParametrizedMagneticFieldProducer.label = "ParabolicMf"

