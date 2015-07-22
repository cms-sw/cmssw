# Load the 3.8 T field map, with the geometry and configuration specified in the GT

from Configuration.StandardSequences.MagneticField_cff import *

VolumeBasedMagneticFieldESProducer.valueOverride = 18268
ParabolicParametrizedMagneticFieldProducer.valueOverride = 18268


