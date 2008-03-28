import FWCore.ParameterSet.Config as cms

#
# Survey&LAS&Cosmics misalignment scenario
# 
# Include this file to produce a misaligned tracker geometry
#
# This replaces Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi
#
# ----------------------------------------
# ----------------- NOTE -----------------
# Sufficient studies do not yet exist to
# provide a reliable version of this 
# scenario (Survey+LAS+Cosmics alignment).
#
# This scenario is not supposed to be used
# to make public(?) estimates of the 
# performance of the CMS.  
#
# This scenario contains lots of guesses,
# especially concerning the improvement
# one can reach by using Cosmics in
# track based alignment.
# The guess is that with Cosmics, one 
# can reach for largest barrel-like parts
# the average alignment accuracy of the 
# 10pb-1 and the SurveyLASOnly scenarios.
#
# The same applies also,but to a lesser
# extent, to the 10pb-1 scenario.
# ------------- NOTE ends ----------------
# ----------------------------------------
from Alignment.TrackerAlignment.Scenarios_cff import *
MisalignedTracker = cms.ESProducer("MisalignedTrackerESProducer",
    TrackerSurveyLASCosmicsScenario
)


