import FWCore.ParameterSet.Config as cms

## To set up NN CPE for track refitting, do:
#from RecoLocalTracker.SiPixelRecHits.customizePixelCPENN import customizePixelCPENN
## Use the centrally installed default model:
#process = customizePixelCPENN(process)

## Select another installed or local model:
#process = customizePixelCPENN(process, "<CMSSW-relative model path>")

_defaultModelPath = "RecoLocalTracker/SiPixelRecHits/data/PixelCPENN/PixelCPENNReco_Run3_2025_v1.onnx"
def customizePixelCPENN(process, modelPath=_defaultModelPath):
    # Start from the PixelCPEGeneric configuration of this process, so that the generic
    # fallback (FPIX, failed NN inference) is the same as the standard generic CPE
    if hasattr(process, "PixelCPEGenericESProducer"):
        genericParams = process.PixelCPEGenericESProducer.parameters_()
    else:
        from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import PixelCPEGenericESProducer
        genericParams = PixelCPEGenericESProducer.parameters_()
    genericParams["ComponentName"] = cms.string("PixelCPENNReco")
    genericParams["modelPath"] = cms.FileInPath(modelPath)

    process.PixelCPENNReco = cms.ESProducer(
        "PixelCPENNRecoESProducer",
        **genericParams
    )

    if not hasattr(process, "TTRHBuilderAngleAndTemplate"):
        process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi")
    process.TTRHBuilderAngleAndTemplate.PixelCPE = "PixelCPENNReco"
    process.TTRHBuilderAngleAndTemplateWithoutProbQ.PixelCPE = "PixelCPENNReco"
    return process
