import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.IterativeFirstTrackFilter_cff import *
from FastSimulation.Tracking.IterativeSecondTrackFilter_cff import *
from FastSimulation.Tracking.IterativeThirdTrackFilter_cff import *
iterativeTrackFiltering = cms.Sequence(iterativeFirstTrackFiltering+iterativeSecondTrackFiltering+iterativeThirdTrackFiltering)

