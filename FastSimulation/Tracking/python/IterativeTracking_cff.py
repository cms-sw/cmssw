import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.PixelTracksProducer_cff import *
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
from FastSimulation.Tracking.IterativeFirstTracking_cff import *
from FastSimulation.Tracking.IterativeSecondTracking_cff import *
from FastSimulation.Tracking.IterativeThirdTracking_cff import *
from FastSimulation.Tracking.IterativeFourthTracking_cff import *
from FastSimulation.Tracking.IterativeFifthTracking_cff import *
from FastSimulation.Tracking.GeneralTracks_cfi import *
from TrackingTools.TrackFitters.TrackFitters_cff import *
from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
iterativeTracking = cms.Sequence(pixelTracking+pixelVertexing
                                 +iterativeFirstTracking
                                 +iterativeSecondTracking
                                 +iterativeThirdTracking
                                 +iterativeFourthTracking
                                 +iterativeFifthTracking
                                 +generalTracks
                                 +trackExtrapolator)
