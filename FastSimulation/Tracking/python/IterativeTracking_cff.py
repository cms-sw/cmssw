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

# this block is to switch between defaul behaviour (MixingMode=='GenMixing') and new mixing
from FastSimulation.Configuration.CommonInputs_cff import MixingMode
if (MixingMode=='DigiRecoMixing'):
#    generalTracksBeforeMixing = FastSimulation.Tracking.GeneralTracks_cfi.generalTracks.clone()
    trackExtrapolator.trackSrc = cms.InputTag("generalTracksBeforeMixing")
    lastTrackingSteps = cms.Sequence(generalTracksBeforeMixing+trackExtrapolator)
elif (MixingMode=='GenMixing'):
    lastTrackingSteps = cms.Sequence(generalTracks+trackExtrapolator)
else:
    print 'unsupported MixingMode label'
        
import RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi
MeasurementTrackerEvent = RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi.MeasurementTrackerEvent.clone(
    pixelClusterProducer = '',
    stripClusterProducer = '',
    inactivePixelDetectorLabels = cms.VInputTag(),
    inactiveStripDetectorLabels = cms.VInputTag(),
    switchOffPixelsIfEmpty = False
)
iterativeTracking = cms.Sequence(pixelTracking+pixelVertexing
                                 +MeasurementTrackerEvent 
                                 +iterativeInitialStep
                                 +iterativeLowPtTripletStep
                                 +iterativePixelPairStep
                                 +iterativeDetachedTripletStep
                                 +iterativeMixedTripletStep
                                 +iterativePixelLessStep
                                 +iterativeTobTecStep
# not validated yet:                                 +muonSeededStep 
#                                 +generalTracks
#                                 +trackExtrapolator)
                                 +lastTrackingSteps)

