import FWCore.ParameterSet.Config as cms

#
# Master configuration for the magnetic field
# old mapping with 4T
from Configuration.StandardSequences.MagneticField_38T_cff import *

# Parabolic parametrized magnetic field used for track building
from MagneticField.ParametrizedEngine.ParabolicParametrizedField_cfi import ParametrizedMagneticFieldProducer as ParabolicParametrizedMagneticFieldProducer
ParabolicParametrizedMagneticFieldProducer.label = "ParabolicMf"
