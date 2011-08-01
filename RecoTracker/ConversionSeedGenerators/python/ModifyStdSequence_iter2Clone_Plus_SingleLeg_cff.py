import FWCore.ParameterSet.Config as cms

from RecoTracker.ConversionSeedGenerators.PhotonConversionTrajectorySeedProducerFromSingleLeg_cff        import *
from RecoTracker.ConversionSeedGenerators.ConversionSequences_iter2Clone_cff        import *
from Configuration.StandardSequences.Reconstruction_cff                  import *

#---- Replace the Input Tags
fifthFilter.recTracks = cms.InputTag("convStep")
#del sixthClusters.oldClusterRemovalInfo
sixthClusters.oldClusterRemovalInfo = cms.InputTag("convClusters")
sixthClusters.trajectories = cms.InputTag("convStep")
sixthClusters.pixelClusters = cms.InputTag("convClusters")
sixthClusters.stripClusters = cms.InputTag("convClusters")
                               

mergeConversionTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'convStep',
    TrackProducer2 = 'sixthStep',
    promoteTrackQuality = True
    )

#---- Define the new generalTracks collection 
generalTracks.TrackProducer1 = 'TrackRefitterStd'
generalTracks.TrackProducer2 = 'mergeConversionTracks'
generalTracks.copyExtras= cms.untracked.bool(False)
generalTracks.makeReKeyedSeeds = cms.untracked.bool(False)

#---- Define the finalTrackCollection
finaltrackCollectionMerging  = cms.Sequence(
    convSequence
    *conversionStepSixth
    * mergeConversionTracks
    * generalTracks)






