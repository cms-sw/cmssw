import FWCore.ParameterSet.Config as cms

#
# Example misalignment scenario
# 
# Include this file to produce a misaligned tracker geometry
#
# This replaces Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi
from Alignment.TrackerAlignment.Scenarios_cff import *
MisalignedTracker = cms.ESProducer("MisalignedTrackerESProducer",
    TrackerExampleScenario
)


