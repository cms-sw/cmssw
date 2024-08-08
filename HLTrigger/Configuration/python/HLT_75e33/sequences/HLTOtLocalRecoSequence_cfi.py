import FWCore.ParameterSet.Config as cms

from ..modules.hltMeasurementTrackerEvent_cfi import *

HLTOtLocalRecoSequence = cms.Sequence(hltMeasurementTrackerEvent)
