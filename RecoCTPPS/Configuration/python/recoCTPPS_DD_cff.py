import FWCore.ParameterSet.Config as cms

from Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi import *
#from Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi import *

from recoCTPPS_sequences_cff import *

recoCTPPS = cms.Sequence(recoCTPPSdets)

