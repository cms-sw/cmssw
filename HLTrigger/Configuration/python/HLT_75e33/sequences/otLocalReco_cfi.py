import FWCore.ParameterSet.Config as cms

from ..modules.bunchSpacingProducer_cfi import *
from ..modules.MeasurementTrackerEvent_cfi import *

otLocalReco = cms.Sequence(MeasurementTrackerEvent+bunchSpacingProducer)
