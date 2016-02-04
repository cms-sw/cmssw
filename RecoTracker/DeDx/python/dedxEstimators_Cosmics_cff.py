import FWCore.ParameterSet.Config as cms

from RecoTracker.DeDx.dedxTruncated40_cfi import *
from RecoTracker.DeDx.dedxMedian_cfi import *
from RecoTracker.DeDx.dedxHarmonic2_cfi import *

dedxTruncated40CTF                                 = dedxTruncated40.clone()
dedxTruncated40CTF.tracks                          = cms.InputTag("ctfWithMaterialTracksP5")
dedxTruncated40CTF.trajectoryTrackAssociation      = cms.InputTag("ctfWithMaterialTracksP5")

dedxHarmonic2CTF                                   = dedxHarmonic2.clone()
dedxHarmonic2CTF.tracks                            = cms.InputTag("ctfWithMaterialTracksP5")
dedxHarmonic2CTF.trajectoryTrackAssociation        = cms.InputTag("ctfWithMaterialTracksP5")

dedxMedianCTF                                      = dedxMedian.clone()
dedxMedianCTF.tracks                               = cms.InputTag("ctfWithMaterialTracksP5")
dedxMedianCTF.trajectoryTrackAssociation           = cms.InputTag("ctfWithMaterialTracksP5")

###RS
dedxTruncated40RS                                  = dedxTruncated40.clone()
dedxTruncated40RS.tracks                           = cms.InputTag("rsWithMaterialTracksP5")
dedxTruncated40RS.trajectoryTrackAssociation       = cms.InputTag("rsWithMaterialTracksP5")

dedxHarmonic2RS                                    = dedxHarmonic2.clone()
dedxHarmonic2RS.tracks                             = cms.InputTag("rsWithMaterialTracksP5")
dedxHarmonic2RS.trajectoryTrackAssociation         = cms.InputTag("rsWithMaterialTracksP5")

dedxMedianRS                                       = dedxMedian.clone()
dedxMedianRS.tracks                                = cms.InputTag("rsWithMaterialTracksP5")
dedxMedianRS.trajectoryTrackAssociation            = cms.InputTag("rsWithMaterialTracksP5")

#TF
dedxTruncated40CosmicTF                            = dedxTruncated40.clone()
dedxTruncated40CosmicTF.tracks                     = cms.InputTag("cosmictrackfinderP5")
dedxTruncated40CosmicTF.trajectoryTrackAssociation = cms.InputTag("cosmictrackfinderP5")

dedxHarmonic2CosmicTF                              = dedxHarmonic2.clone()
dedxHarmonic2CosmicTF.tracks                       = cms.InputTag("cosmictrackfinderP5")
dedxHarmonic2CosmicTF.trajectoryTrackAssociation   = cms.InputTag("cosmictrackfinderP5")

dedxMedianCosmicTF                                 = dedxMedian.clone()
dedxMedianCosmicTF.tracks                          = cms.InputTag("cosmictrackfinderP5")
dedxMedianCosmicTF.trajectoryTrackAssociation      = cms.InputTag("cosmictrackfinderP5")

#CTF P5 LHC
dedxTruncated40CTFP5LHC                            = dedxTruncated40.clone()
dedxTruncated40CTFP5LHC.tracks                     = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")
dedxTruncated40CTFP5LHC.trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")

dedxMedianCTFP5LHC                                 = dedxMedian.clone()
dedxMedianCTFP5LHC.tracks                          = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")
dedxMedianCTFP5LHC.trajectoryTrackAssociation      = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")

dedxHarmonic2CTFP5LHC                              = dedxHarmonic2.clone()
dedxHarmonic2CTFP5LHC.tracks                       = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")
dedxHarmonic2CTFP5LHC.trajectoryTrackAssociation   = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")


doAlldEdXEstimatorsCTF      = cms.Sequence(dedxTruncated40CTF      + dedxMedianCTF      + dedxHarmonic2CTF)
doAlldEdXEstimatorsRS       = cms.Sequence(dedxTruncated40RS       + dedxMedianRS       + dedxHarmonic2RS)
doAlldEdXEstimatorsCosmicTF = cms.Sequence(dedxTruncated40CosmicTF + dedxMedianCosmicTF + dedxHarmonic2CosmicTF)
doAlldEdXEstimatorsCTFP5LHC = cms.Sequence(dedxTruncated40CTFP5LHC + dedxMedianCTFP5LHC + dedxHarmonic2CTFP5LHC)


doAllCosmicdEdXEstimators = cms.Sequence( doAlldEdXEstimatorsCTF + doAlldEdXEstimatorsRS + doAlldEdXEstimatorsCosmicTF + doAlldEdXEstimatorsCTFP5LHC)

