import FWCore.ParameterSet.Config as cms

from ..modules.MeasurementTrackerEvent_cfi import *

otLocalRecoSequence = cms.Sequence(MeasurementTrackerEvent)
