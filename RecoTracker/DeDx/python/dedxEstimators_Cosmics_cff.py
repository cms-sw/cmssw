import FWCore.ParameterSet.Config as cms

#from RecoTracker.DeDx.dedxEstimatorsFromRefitter_cff import *

from RecoTracker.DeDx.dedxTruncated40_cfi import *
from RecoTracker.DeDx.dedxMedian_cfi import *
from RecoTracker.DeDx.dedxHarmonic2_cfi import *
from RecoTracker.DeDx.dedxUnbinned_cfi import *

###CTF
dedxTruncated40CTF = dedxTruncated40.clone()
dedxTruncated40CTF.tracks=cms.InputTag("ctfWithMaterialTracksP5")
dedxTruncated40CTF.trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5")

dedxHarmonic2CTF=dedxHarmonic2.clone()
dedxHarmonic2CTF.tracks=cms.InputTag("ctfWithMaterialTracksP5")
dedxHarmonic2CTF.trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5")

dedxMedianCTF=dedxMedian.clone()
dedxMedianCTF.tracks=cms.InputTag("ctfWithMaterialTracksP5")
dedxMedianCTF.trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5")

dedxUnbinnedCTF=dedxUnbinned.clone()
dedxUnbinnedCTF.tracks=cms.InputTag("ctfWithMaterialTracksP5")
dedxUnbinnedCTF.trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5")


###RS
dedxTruncated40RS = dedxTruncated40.clone()
dedxTruncated40RS.tracks=cms.InputTag("rsWithMaterialTracksP5")
dedxTruncated40RS.trajectoryTrackAssociation = cms.InputTag("rsWithMaterialTracksP5")

dedxHarmonic2RS=dedxHarmonic2.clone()
dedxHarmonic2RS.tracks=cms.InputTag("rsWithMaterialTracksP5")
dedxHarmonic2RS.trajectoryTrackAssociation = cms.InputTag("rsWithMaterialTracksP5")

dedxMedianRS=dedxMedian.clone()
dedxMedianRS.tracks=cms.InputTag("rsWithMaterialTracksP5")
dedxMedianRS.trajectoryTrackAssociation = cms.InputTag("rsWithMaterialTracksP5")

dedxUnbinnedRS=dedxUnbinned.clone()
dedxUnbinnedRS.tracks=cms.InputTag("rsWithMaterialTracksP5")
dedxUnbinnedRS.trajectoryTrackAssociation = cms.InputTag("rsWithMaterialTracksP5")

###CosmicTF
dedxTruncated40CosmicTF = dedxTruncated40.clone()
dedxTruncated40CosmicTF.tracks=cms.InputTag("cosmictrackfinderP5")
dedxTruncated40CosmicTF.trajectoryTrackAssociation = cms.InputTag("cosmictrackfinderP5")

dedxHarmonic2CosmicTF=dedxHarmonic2.clone()
dedxHarmonic2CosmicTF.tracks=cms.InputTag("cosmictrackfinderP5")
dedxHarmonic2CosmicTF.trajectoryTrackAssociation = cms.InputTag("cosmictrackfinderP5")

dedxMedianCosmicTF=dedxMedian.clone()
dedxMedianCosmicTF.tracks=cms.InputTag("cosmictrackfinderP5")
dedxMedianCosmicTF.trajectoryTrackAssociation = cms.InputTag("cosmictrackfinderP5")

dedxUnbinnedCosmicTF=dedxUnbinned.clone()
dedxUnbinnedCosmicTF.tracks=cms.InputTag("cosmictrackfinderP5")
dedxUnbinnedCosmicTF.trajectoryTrackAssociation = cms.InputTag("cosmictrackfinderP5")


doAlldEdXEstimatorsCTF = cms.Sequence(dedxTruncated40CTF + dedxMedianCTF + dedxHarmonic2CTF + dedxUnbinnedCTF)

doAlldEdXEstimatorsRS = cms.Sequence(dedxTruncated40RS + dedxMedianRS + dedxHarmonic2RS + dedxUnbinnedRS)

doAlldEdXEstimatorsCosmicTF = cms.Sequence(dedxTruncated40CosmicTF + dedxMedianCosmicTF + dedxHarmonic2CosmicTF +  dedxUnbinnedCosmicTF)


doAlldEdXEstimators = cms.Sequence( doAlldEdXEstimatorsCTF + doAlldEdXEstimatorsRS + doAlldEdXEstimatorsCosmicTF )

