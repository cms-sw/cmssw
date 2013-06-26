import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use the VolumeBased magnetic
# field engine.
#Default is version 85l
from MagneticField.Engine.volumeBasedMagneticField_1103l_cfi import *
VolumeBasedMagneticFieldESProducer.version = 'grid_1103l_071212_4t'
ParametrizedMagneticFieldProducer.parameters.BValue = '4_0T'


