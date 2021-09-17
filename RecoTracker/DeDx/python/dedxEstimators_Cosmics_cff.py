import FWCore.ParameterSet.Config as cms


from RecoTracker.DeDx.dedxEstimators_cff import *

dedxHitInfoCTF     = dedxHitInfo.clone( tracks = "ctfWithMaterialTracksP5")

dedxTruncated40CTF = dedxTruncated40.clone( tracks = "ctfWithMaterialTracksP5")

dedxHarmonic2CTF   = dedxHarmonic2.clone( tracks = "ctfWithMaterialTracksP5")

dedxDiscrimASmiCTF = dedxDiscrimASmi.clone( tracks = "ctfWithMaterialTracksP5")

#TF
dedxHitInfoCosmicTF     = dedxHitInfo.clone( tracks = "cosmictrackfinderP5")

dedxTruncated40CosmicTF = dedxTruncated40.clone( tracks = "cosmictrackfinderP5")

dedxHarmonic2CosmicTF   = dedxHarmonic2.clone( tracks = "cosmictrackfinderP5")

dedxDiscrimASmiCosmicTF = dedxDiscrimASmi.clone( tracks = "cosmictrackfinderP5")

#CTF P5 LHC
dedxHitInfoCTFP5LHC     = dedxHitInfo.clone( tracks = "ctfWithMaterialTracksP5LHCNavigation")

dedxTruncated40CTFP5LHC = dedxTruncated40.clone( tracks = "ctfWithMaterialTracksP5LHCNavigation")

dedxDiscrimASmiCTFP5LHC = dedxDiscrimASmi.clone( tracks = "ctfWithMaterialTracksP5LHCNavigation")

dedxHarmonic2CTFP5LHC   = dedxHarmonic2.clone( tracks = "ctfWithMaterialTracksP5LHCNavigation")

doAlldEdXEstimatorsCTFTask      = cms.Task(dedxTruncated40CTF      , dedxHitInfoCTF      , dedxHarmonic2CTF)
doAlldEdXEstimatorsCTF      = cms.Sequence(doAlldEdXEstimatorsCTFTask)
doAlldEdXEstimatorsCosmicTFTask = cms.Task(dedxTruncated40CosmicTF , dedxHitInfoCosmicTF , dedxHarmonic2CosmicTF)
doAlldEdXEstimatorsCosmicTF = cms.Sequence(doAlldEdXEstimatorsCosmicTFTask)
doAlldEdXEstimatorsCTFP5LHCTask = cms.Task(dedxTruncated40CTFP5LHC , dedxHitInfoCTFP5LHC , dedxHarmonic2CTFP5LHC)
doAlldEdXEstimatorsCTFP5LHC = cms.Sequence(doAlldEdXEstimatorsCTFP5LHCTask)

doAllCosmicdEdXEstimatorsTask = cms.Task( doAlldEdXEstimatorsCTFTask , doAlldEdXEstimatorsCosmicTFTask , doAlldEdXEstimatorsCTFP5LHCTask)
doAllCosmicdEdXEstimators = cms.Sequence(doAllCosmicdEdXEstimatorsTask)
