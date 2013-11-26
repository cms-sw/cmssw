import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.PixelTracksProducer_cff import *
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
from TrackingTools.TrackFitters.TrackFitters_cff import *
from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *

from FastSimulation.Tracking.IterativeInitialStep_Phase2_cff import *
from FastSimulation.Tracking.IterativeSecondStep_Phase2_cff import *
from FastSimulation.Tracking.GeneralTracks_Phase2_cfi import *

iterativeTracking = cms.Sequence(pixelTracking+pixelVertexing
                                 +iterativeInitialStep
                                 +iterativeSecondStep
                                 +generalTracks
                                 +trackExtrapolator)


