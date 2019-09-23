import FWCore.ParameterSet.Config as cms


from RecoTracker.DeDx.dedxEstimators_cff import *

dedxHitInfoCTF                                     = dedxHitInfo.clone()
dedxHitInfoCTF.tracks                              = cms.InputTag("ctfWithMaterialTracksP5")

dedxTruncated40CTF                                 = dedxTruncated40.clone()
dedxTruncated40CTF.tracks                          = cms.InputTag("ctfWithMaterialTracksP5")

dedxHarmonic2CTF                                   = dedxHarmonic2.clone()
dedxHarmonic2CTF.tracks                            = cms.InputTag("ctfWithMaterialTracksP5")

dedxDiscrimASmiCTF                                      = dedxDiscrimASmi.clone()
dedxDiscrimASmiCTF.tracks                               = cms.InputTag("ctfWithMaterialTracksP5")

#TF
dedxHitInfoCosmicTF                                = dedxHitInfo.clone()
dedxHitInfoCosmicTF.tracks                         = cms.InputTag("cosmictrackfinderP5")

dedxTruncated40CosmicTF                            = dedxTruncated40.clone()
dedxTruncated40CosmicTF.tracks                     = cms.InputTag("cosmictrackfinderP5")

dedxHarmonic2CosmicTF                              = dedxHarmonic2.clone()
dedxHarmonic2CosmicTF.tracks                       = cms.InputTag("cosmictrackfinderP5")

dedxDiscrimASmiCosmicTF                                 = dedxDiscrimASmi.clone()
dedxDiscrimASmiCosmicTF.tracks                          = cms.InputTag("cosmictrackfinderP5")

#CTF P5 LHC
dedxHitInfoCTFP5LHC                                = dedxHitInfo.clone()
dedxHitInfoCTFP5LHC.tracks                         = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")

dedxTruncated40CTFP5LHC                            = dedxTruncated40.clone()
dedxTruncated40CTFP5LHC.tracks                     = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")

dedxDiscrimASmiCTFP5LHC                                 = dedxDiscrimASmi.clone()
dedxDiscrimASmiCTFP5LHC.tracks                          = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")

dedxHarmonic2CTFP5LHC                              = dedxHarmonic2.clone()
dedxHarmonic2CTFP5LHC.tracks                       = cms.InputTag("ctfWithMaterialTracksP5LHCNavigation")

doAlldEdXEstimatorsCTFTask      = cms.Task(dedxTruncated40CTF      , dedxHitInfoCTF      , dedxHarmonic2CTF)
doAlldEdXEstimatorsCTF      = cms.Sequence(doAlldEdXEstimatorsCTFTask)
doAlldEdXEstimatorsCosmicTFTask = cms.Task(dedxTruncated40CosmicTF , dedxHitInfoCosmicTF , dedxHarmonic2CosmicTF)
doAlldEdXEstimatorsCosmicTF = cms.Sequence(doAlldEdXEstimatorsCosmicTFTask)
doAlldEdXEstimatorsCTFP5LHCTask = cms.Task(dedxTruncated40CTFP5LHC , dedxHitInfoCTFP5LHC , dedxHarmonic2CTFP5LHC)
doAlldEdXEstimatorsCTFP5LHC = cms.Sequence(doAlldEdXEstimatorsCTFP5LHCTask)

doAllCosmicdEdXEstimatorsTask = cms.Task( doAlldEdXEstimatorsCTFTask , doAlldEdXEstimatorsCosmicTFTask , doAlldEdXEstimatorsCTFP5LHCTask)
doAllCosmicdEdXEstimators = cms.Sequence(doAllCosmicdEdXEstimatorsTask)
