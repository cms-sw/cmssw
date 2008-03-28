import FWCore.ParameterSet.Config as cms

#
# 1000 pb-1 misalignment scenario (see the corresponding CMS IN)
# 
# Include this file to produce a misaligned tracker geometry
#
# This replaces Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi
#
from Alignment.TrackerAlignment.Scenarios_cff import *
MisalignedTracker = cms.ESProducer("MisalignedTrackerESProducer",
    Tracker1000pbScenario
)


