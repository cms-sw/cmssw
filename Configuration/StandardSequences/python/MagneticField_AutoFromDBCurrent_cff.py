# Load the field map corresponding to the current stored in the runInfo,
# with the geometry and configuration specified in the GT

import FWCore.ParameterSet.Config as cms
from MagneticField.Engine.volumeBasedMagneticFieldFromDB_cfi import *

# Parabolic parametrized magnetic field used for track building.
# NOTE that as of now  this does not scale with the current from the DB.
from MagneticField.ParametrizedEngine.ParabolicParametrizedField_cfi import ParametrizedMagneticFieldProducer as ParabolicParametrizedMagneticFieldProducer
ParabolicParametrizedMagneticFieldProducer.label = "ParabolicMf"


