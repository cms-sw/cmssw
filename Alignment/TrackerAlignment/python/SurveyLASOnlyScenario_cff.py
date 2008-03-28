import FWCore.ParameterSet.Config as cms

#
# Survey&LAS only misalignment scenario (see table 6 of 
#  CMS IN 2007/036)
# 
# Include this file to produce a misaligned tracker geometry
#
# This replaces Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi
#
from Alignment.TrackerAlignment.Scenarios_cff import *
MisalignedTracker = cms.ESProducer("MisalignedTrackerESProducer",
    TrackerSurveyLASOnlyScenario
)


