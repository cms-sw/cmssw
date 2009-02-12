import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *

from RecoParticleFlow.PFTracking.pfConversions_cfi import *

#TRAJECTORIES IN THE EVENT


#UNCOMMENT THE LINES THAT START WITH #DON# IN ORDER TO ADD CONVERSION FROM PF CLUSTERS
#DON#from RecoEgamma.EgammaPhotonProducers.softConversionSequence_cff import *
#DON#softConversionIOTracks.TrajectoryInEvent = cms.bool(True)
#DON#softConversionOITracks.TrajectoryInEvent = cms.bool(True)
#DON#pfConversions.OtherConversionCollection =cms.VInputTag(cms.InputTag("softConversions:softConversionCollection"))
#DON#pfConversions.OtherOutInCollection      =           cms.VInputTag(cms.InputTag("softConversionOITracks"))
#DON#pfConversions.OtherInOutCollection      =           cms.VInputTag(cms.InputTag("softConversionIOTracks"))

#UNCOMMENT THE LINES THAT START WITH #HON# IN ORDER TO ADD CONVERSION FROM GENERAL TRACKS
#HON#from RecoEgamma.EgammaPhotonProducers.trackerOnlyConversionSequence_cff import *

#HON#pfConversions.OtherConversionCollection =cms.VInputTag(cms.InputTag("trackerOnlyConversions"))
#HON#pfConversions.OtherOutInCollection      =           cms.VInputTag(cms.InputTag("generalTracks"))
#HON#pfConversions.OtherInOutCollection      =           cms.VInputTag(cms.InputTag("generalTracks"))

particleFlowTrackWithConversion =cms.Sequence(
    pfTrackElec*
    #HON#trackerOnlyConversionSequence*
    #DON#    softConversionSequence*
    pfConversions
    )
