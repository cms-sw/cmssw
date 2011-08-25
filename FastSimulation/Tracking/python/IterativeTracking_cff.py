import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.PixelTracksProducer_cff import *
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
#from FastSimulation.Tracking.IterativeFirstTracking_cff import *
from FastSimulation.Tracking.IterativeInitialStep_cff import *
from FastSimulation.Tracking.IterativeLowPtTripletStep_cff import *
from FastSimulation.Tracking.IterativePixelPairStep_cff import *
#from FastSimulation.Tracking.IterativeSecondTracking_cff import *
from FastSimulation.Tracking.IterativeDetachedTripletStep_cff import *
#from FastSimulation.Tracking.IterativeThirdTracking_cff import *
from FastSimulation.Tracking.IterativeMixedTripletStep_cff import *
#from FastSimulation.Tracking.IterativeFourthTracking_cff import *
from FastSimulation.Tracking.IterativePixelLessStep_cff import *
#from FastSimulation.Tracking.IterativeFifthTracking_cff import *
from FastSimulation.Tracking.IterativeTobTecStep_cff import *
from FastSimulation.Tracking.GeneralTracks_cfi import *
from TrackingTools.TrackFitters.TrackFitters_cff import *
from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
iterativeTracking = cms.Sequence(pixelTracking+pixelVertexing
#                                 +iterativeFirstTracking
                                 +iterativeInitialStep
                                 +iterativeLowPtTripletStep
                                 +iterativePixelPairStep
#                                 +iterativeSecondTracking
                                 +iterativeDetachedTripletStep
#                                 +iterativeThirdTracking
                                 +iterativeMixedTripletStep
#                                 +iterativeFourthTracking
                                 +iterativePixelLessStep
#                                 +iterativeFifthTracking
                                 +iterativeTobTecStep
                                 +generalTracks
                                 +trackExtrapolator)

