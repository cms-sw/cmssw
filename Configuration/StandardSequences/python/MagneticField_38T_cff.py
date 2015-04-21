# Load the 3.8 T field map, with the geometry and configuration specified in the GT

import FWCore.ParameterSet.Config as cms
from MagneticField.Engine.volumeBasedMagneticFieldFromDB_cfi import *
VolumeBasedMagneticFieldESProducer.valueOverride = 18268
