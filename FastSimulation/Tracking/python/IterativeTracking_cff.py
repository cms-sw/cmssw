import FWCore.ParameterSet.Config as cms

whichTracking = 'old' # 'old' is the default for the moment

from FastSimulation.Tracking.PixelTracksProducer_cff import *
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
from FastSimulation.Tracking.GeneralTracks_cfi import *
from TrackingTools.TrackFitters.TrackFitters_cff import *
from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *

if(whichTracking=='old'):
#### OLD iterative tracking (temporarily remains the default)
    from FastSimulation.Tracking.IterativeFirstTracking_cff import *
    from FastSimulation.Tracking.IterativeSecondTracking_cff import *
    from FastSimulation.Tracking.IterativeThirdTracking_cff import *
    from FastSimulation.Tracking.IterativeFourthTracking_cff import *
    from FastSimulation.Tracking.IterativeFifthTracking_cff import *
    iterativeTracking = cms.Sequence(pixelTracking+pixelVertexing 
                                     +iterativeFirstTracking
                                     +iterativeSecondTracking
                                     +iterativeThirdTracking
                                     +iterativeFourthTracking
                                     +iterativeFifthTracking
                                     +generalTracks
                                     +trackExtrapolator)
else:
#### NEW iterative tracking
    from FastSimulation.Tracking.IterativeInitialStep_cff import *
    from FastSimulation.Tracking.IterativeLowPtTripletStep_cff import *
    from FastSimulation.Tracking.IterativePixelPairStep_cff import *
    from FastSimulation.Tracking.IterativeDetachedTripletStep_cff import *
    from FastSimulation.Tracking.IterativeMixedTripletStep_cff import *
    from FastSimulation.Tracking.IterativePixelLessStep_cff import *
    from FastSimulation.Tracking.IterativeTobTecStep_cff import *
    iterativeTracking = cms.Sequence(pixelTracking+pixelVertexing
                                     +iterativeInitialStep
                                     +iterativeLowPtTripletStep
                                     +iterativePixelPairStep
                                     +iterativeDetachedTripletStep
                                     +iterativeMixedTripletStep
                                     +iterativePixelLessStep
                                     +iterativeTobTecStep
                                     +generalTracks
                                     +trackExtrapolator)


