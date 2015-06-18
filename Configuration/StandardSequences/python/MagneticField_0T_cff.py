# Load the 0T field map, with the geometry specified in the GT

from Configuration.StandardSequences.MagneticField_cff import *

VolumeBasedMagneticFieldESProducer.valueOverride = 0
ParabolicParametrizedMagneticFieldProducer.valueOverride = 0
