import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use the VolumeBased magnetic
# field engine.
from MagneticField.Engine.volumeBasedMagneticField_090322_2pi_scaled_cfi import *


ParametrizedMagneticFieldProducer.version = 'PolyFit2D'
ParametrizedMagneticFieldProducer.parameters = cms.PSet(
    BValue = cms.double(3.81143026675623)
) 
