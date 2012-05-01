import FWCore.ParameterSet.Config as cms

#from FastSimulation.Tracking.PixelTracksProducer_cff import *
#from FastSimulation.Tracking.PixelVerticesProducer_cff import *
#from FastSimulation.Tracking.IterativeFirstTracking_cff import *
#from FastSimulation.Tracking.IterativeSecondTracking_cff import *
#from FastSimulation.Tracking.IterativeThirdTracking_cff import *
#from FastSimulation.Tracking.IterativeFourthTracking_cff import *
####from FastSimulation.Tracking.IterativeFifthTracking_cff import *
from FastSimulation.Tracking.HLTGeneralTracks_cfi import *
#from TrackingTools.TrackFitters.TrackFitters_cff import *
#from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *

hltIter4Merged = HLTgeneralTracks.clone()
#hltPFJetCtfWithMaterialTracks = HLTgeneralTracks.clone()


HLTIterativeTracking = cms.Sequence(hltIter4Merged
                                    )
