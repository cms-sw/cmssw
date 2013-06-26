import FWCore.ParameterSet.Config as cms

# Include this file to produce a misaligned tracker geometry
#
import Alignment.TrackerAlignment.Scenarios_cff as Scenarios
# This replaces Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi
MisalignedTracker = cms.ESProducer("MisalignedTrackerESProducer",
                                   saveToDbase = cms.untracked.bool(False),
                                   saveFakeScenario = cms.untracked.bool(False),
                                   scenario = Scenarios.NoMovementsScenario # a cms.PSet
                                   )


