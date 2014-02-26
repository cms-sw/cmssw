import FWCore.ParameterSet.Config as cms

#this will load the auto magnetic field producer reading the current from the DB
# and loading the best map available for that current as specified in the file 
from MagneticField.Engine.autoMagneticFieldProducer_cfi import *

# Parabolic parametrized magnetic field used for track building
from MagneticField.ParametrizedEngine.ParabolicParametrizedField_cfi import ParametrizedMagneticFieldProducer as ParabolicParametrizedMagneticFieldProducer
ParabolicParametrizedMagneticFieldProducer.label = "ParabolicMf"


