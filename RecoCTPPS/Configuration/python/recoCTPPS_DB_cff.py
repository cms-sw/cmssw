import FWCore.ParameterSet.Config as cms

from Geometry.VeryForwardGeometry.geometryRPFromDB_cfi import *

from recoCTPPS_sequences_cff import *

recoCTPPS = cms.Sequence(recoCTPPSdets)
