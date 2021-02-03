import FWCore.ParameterSet.Config as cms

from ..modules.MeasurementTrackerEvent_cfi import *

otLocalReco = cms.Sequence(MeasurementTrackerEvent)
