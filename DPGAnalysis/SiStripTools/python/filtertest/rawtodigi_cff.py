import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

# Magnetic fiuld: force mag field to be 3.8 tesla
from Configuration.StandardSequences.MagneticField_38T_cff import *

#Geometry
from Configuration.StandardSequences.GeometryRecoDB_cff import *

# Real data raw to digi
from Configuration.StandardSequences.RawToDigi_Data_cff import *

from Configuration.StandardSequences.ReconstructionCosmics_cff import *
