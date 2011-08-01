import FWCore.ParameterSet.Config as cms

from RecoTracker.ConversionSeedGenerators.ConversionSequences_cff        import *
from Configuration.StandardSequences.Reconstruction_cff                  import *

#---- Track Refitter needed to create Trajectories, to be used to generate the strip/pixel cluster collections for the new iter steps  
from RecoTracker.TrackProducer.TrackRefitters_cff                        import *
TrackRefitterStd = TrackRefitter.clone()

#---- Replace the Input Tags
del sixthClusters.oldClusterRemovalInfo
sixthClusters.trajectories = cms.InputTag("TrackRefitterStd")
sixthClusters.pixelClusters = cms.InputTag("siPixelClusters")
sixthClusters.stripClusters = cms.InputTag("siStripClusters")
                               

#sixthClusters.remove(oldClusterRemovalInfo)
fifthFilter.recTracks = cms.InputTag("TrackRefitterStd")
seventhSeedsTk.inputTrajectory   = cms.InputTag("TrackRefitterStd")

#---- Merge the Conversion collections
mergeConversionTracks = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'sixthStep',
    TrackProducer2 = 'seventhStep',
    promoteTrackQuality = True
    )

#---- Define the new generalTracks collection 
generalTracks.TrackProducer1 = 'TrackRefitterStd'
generalTracks.TrackProducer2 = 'mergeConversionTracks'
generalTracks.copyExtras= cms.untracked.bool(False)
generalTracks.makeReKeyedSeeds = cms.untracked.bool(False)

#---- Define the finalTrackCollection
finaltrackCollectionMerging  = cms.Sequence(TrackRefitterStd * conversionStep * mergeConversionTracks * generalTracks)







