import FWCore.ParameterSet.Config as cms

from ..modules.MeasurementTrackerEvent_cfi import *

otLocalRecoTask = cms.Task(
    MeasurementTrackerEvent
)
