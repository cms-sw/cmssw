import FWCore.ParameterSet.Config as cms

#
# "Misalignment" scenario without misalignment...
# 
# Include this file to produce an aligned tracker geometry with the misalignment tools
#
# This replaces Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi
from Alignment.TrackerAlignment.Scenarios_cff import *
MisalignedTracker = cms.ESProducer("MisalignedTrackerESProducer",
    NoMovementsScenario
)


