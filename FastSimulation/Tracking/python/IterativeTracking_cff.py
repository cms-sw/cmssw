import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.PixelTracksProducer_cff import *
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
from FastSimulation.Tracking.IterativeTrackSeedProducer_cff import *
from FastSimulation.Tracking.IterativeCandidateProducer_cff import *
from FastSimulation.Tracking.IterativeTrackProducer_cff import *
from FastSimulation.Tracking.IterativeTrackMerger_cff import *
from FastSimulation.Tracking.IterativeTrackFilter_cff import *
iterativeTracking = cms.Sequence(pixelGSTracking+pixelGSVertexing+iterativeTrackingSeeds+iterativeTrackCandidates+iterativeTrackFitting+iterativeTrackMerging+iterativeTrackFiltering)

