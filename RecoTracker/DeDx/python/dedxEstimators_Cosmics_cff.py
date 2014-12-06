import FWCore.ParameterSet.Config as cms


from RecoTracker.DeDx.dedxEstimators_cff import *

dedxHitInfoCTF                                     = dedxHitInfo.clone()
dedxHitInfoCTF.tracks                              = cms.InputTag("ctfWithMaterialTracksP5")
dedxHitInfoCTF.trajectoryTrackAssociation          = cms.InputTag("ctfWithMaterialTracksP5")

dedxTruncated40CTF                                 = dedxTruncated40.clone()
dedxTruncated40CTF.tracks                          = cms.InputTag("ctfWithMaterialTracksP5")
dedxTruncated40CTF.trajectoryTrackAssociation      = cms.InputTag("ctfWithMaterialTracksP5")

dedxHarmonic2CTF                                   = dedxHarmonic2.clone()
dedxHarmonic2CTF.tracks                            = cms.InputTag("ctfWithMaterialTracksP5")
dedxHarmonic2CTF.trajectoryTrackAssociation        = cms.InputTag("ctfWithMaterialTracksP5")

dedxDiscrimASmiCTF                                      = dedxDiscrimASmi.clone()
dedxDiscrimASmiCTF.tracks                               = cms.InputTag("ctfWithMaterialTracksP5")
dedxDiscrimASmiCTF.trajectoryTrackAssociation           = cms.InputTag("ctfWithMaterialTracksP5")

#TF
dedxHitInfoCosmicTF                                = dedxHitInfo.clone()
dedxHitInfoCosmicTF.tracks                         = cms.InputTag("cosmictrackfinderP5")
dedxHitInfoCosmicTF.trajectoryTrackAssociation     = cms.InputTag("cosmictrackfinderP5")

dedxTruncated40CosmicTF                            = dedxTruncated40.clone()
dedxTruncated40CosmicTF.tracks                     = cms.InputTag("cosmictrackfinderP5")
dedxTruncated40CosmicTF.trajectoryTrackAssociation = cms.InputTag("cosmictrackfinderP5")

dedxHarmonic2CosmicTF                              = dedxHarmonic2.clone()
dedxHarmonic2CosmicTF.tracks                       = cms.InputTag("cosmictrackfinderP5")
dedxHarmonic2CosmicTF.trajectoryTrackAssociation   = cms.InputTag("cosmictrackfinderP5")

dedxDiscrimASmiCosmicTF                                 = dedxDiscrimASmi.clone()
dedxDiscrimASmiCosmicTF.tracks                          = cms.InputTag("cosmictrackfinderP5")
dedxDiscrimASmiCosmicTF.trajectoryTrackAssociation      = cms.InputTag("cosmictrackfinderP5")

#CTF P5 LHC
dedxHitInfoCTFP5LHC                                = dedxHitInfo.clone()
dedxHitInfoCTFP5LHC.tracks                         = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")
dedxHitInfoCTFP5LHC.trajectoryTrackAssociation     = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")

dedxTruncated40CTFP5LHC                            = dedxTruncated40.clone()
dedxTruncated40CTFP5LHC.tracks                     = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")
dedxTruncated40CTFP5LHC.trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")

dedxDiscrimASmiCTFP5LHC                                 = dedxDiscrimASmi.clone()
dedxDiscrimASmiCTFP5LHC.tracks                          = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")
dedxDiscrimASmiCTFP5LHC.trajectoryTrackAssociation      = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")

dedxHarmonic2CTFP5LHC                              = dedxHarmonic2.clone()
dedxHarmonic2CTFP5LHC.tracks                       = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")
dedxHarmonic2CTFP5LHC.trajectoryTrackAssociation   = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")


doAlldEdXEstimatorsCTF      = cms.Sequence(dedxTruncated40CTF      + dedxHitInfoCTF      + dedxHarmonic2CTF)
doAlldEdXEstimatorsCosmicTF = cms.Sequence(dedxTruncated40CosmicTF + dedxHitInfoCosmicTF + dedxHarmonic2CosmicTF)
doAlldEdXEstimatorsCTFP5LHC = cms.Sequence(dedxTruncated40CTFP5LHC + dedxHitInfoCTFP5LHC + dedxHarmonic2CTFP5LHC)

doAllCosmicdEdXEstimators = cms.Sequence( doAlldEdXEstimatorsCTF + doAlldEdXEstimatorsCosmicTF + doAlldEdXEstimatorsCTFP5LHC)

