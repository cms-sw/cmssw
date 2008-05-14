import FWCore.ParameterSet.Config as cms

#---------------------------------------------------
# AlCaReco filtering for the Tracker Laser ALignment
#---------------------------------------------------
# create sequence for rechit filtering for phi symmetry calibration
from Alignment.LaserAlignment.LaserAlignmentT0Producer_cfi import *
seqALCARECOTkAlLAS = cms.Sequence(laserAlignmentT0Producer)

