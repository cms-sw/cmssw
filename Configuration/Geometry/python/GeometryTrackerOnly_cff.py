import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal Tracker only geometry, needed for simulation
from Configuration.Geometry.GeometrySimTracker_cff import *

# Reco Tracker only geometry
from Configuration.Geometry.GeometryRecoTracker_cff import *
