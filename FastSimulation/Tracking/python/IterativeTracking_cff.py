import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.PixelTracksProducer_cff import *
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
from FastSimulation.Tracking.GeneralTracks_cfi import *
from TrackingTools.TrackFitters.TrackFitters_cff import *
from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *

from FastSimulation.Tracking.IterativeInitialStep_cff import *
from FastSimulation.Tracking.IterativeLowPtTripletStep_cff import *
from FastSimulation.Tracking.IterativePixelPairStep_cff import *
from FastSimulation.Tracking.IterativeDetachedTripletStep_cff import *
from FastSimulation.Tracking.IterativeMixedTripletStep_cff import *
from FastSimulation.Tracking.IterativePixelLessStep_cff import *
from FastSimulation.Tracking.IterativeTobTecStep_cff import *
from FastSimulation.Tracking.MuonSeededStep_cff import *

iterativeTracking = cms.Sequence(pixelTracking+pixelVertexing
                                 +iterativeInitialStep
                                 +iterativeLowPtTripletStep
                                 +iterativePixelPairStep
                                 +iterativeDetachedTripletStep
                                 +iterativeMixedTripletStep
                                 +iterativePixelLessStep
                                 +iterativeTobTecStep
# not validated yet:                                 +muonSeededStep 
                                 +generalTracks
                                 +trackExtrapolator)


