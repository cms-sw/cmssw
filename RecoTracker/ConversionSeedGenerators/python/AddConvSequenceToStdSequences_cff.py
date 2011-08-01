import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Reconstruction_cff                    import *
from RecoTracker.ConversionSeedGenerators.PhotonConversionTrajectorySeedProducerFromSingleLeg_cff import *

tmpgeneralTracks=generalTracks.clone()
trackCollectionMerging.remove(generalTracks)
trackCollectionMerging *= tmpgeneralTracks
trackCollectionMerging *= generalTracks
generalTracks.TrackProducer1 = 'tmpgeneralTracks'
generalTracks.TrackProducer2 = 'convStep'

