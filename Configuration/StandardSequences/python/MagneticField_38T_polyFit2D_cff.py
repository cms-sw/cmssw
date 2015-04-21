#Configuration for teh RUN 1 field map + detailed (2D) parametrization of tracker region.

import FWCore.ParameterSet.Config as cms
from MagneticField.Engine.volumeBasedMagneticField_090322_2pi_scaled_cfi import *


ParametrizedMagneticFieldProducer.version = 'PolyFit2D'
ParametrizedMagneticFieldProducer.parameters = cms.PSet(
    BValue = cms.double(3.81143026675623)
) 
