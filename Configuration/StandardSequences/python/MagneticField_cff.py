# Load the default 3.8 T field map with the geometry and configuration specified in the GT.
# Note that this does NOT depend on the actual solenoid current:
# MagneticField_AutoFromDBCurrent_cff should be used if a current-dependent map is needed.
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.MagneticField_38T_cff import *
