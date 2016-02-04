import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.IterativeFirstTrackProducer_cff import *
from FastSimulation.Tracking.IterativeSecondTrackProducer_cff import *
from FastSimulation.Tracking.IterativeThirdTrackProducer_cff import *
iterativeTrackFitting = cms.Sequence(iterativeFirstTracks+iterativeSecondTracks+iterativeThirdTracks)

