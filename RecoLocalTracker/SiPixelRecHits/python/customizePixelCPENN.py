import FWCore.ParameterSet.Config as cms

# To set up NN CPE for track refitting, do:
#from RecoLocalTracker.SiPixelRecHits.customizePixelCPENN import customizePixelCPENN
#process = customizePixelCPENN(process, "<modelDirectory>")


def customizePixelCPENN(process, modelDirectory):
    # Start from the PixelCPEGeneric configuration of this process, so that the generic
    # fallback (FPIX, failed NN inference) is the same as the standard generic CPE
    if hasattr(process, "PixelCPEGenericESProducer"):
        genericParams = process.PixelCPEGenericESProducer.parameters_()
    else:
        from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import PixelCPEGenericESProducer
        genericParams = PixelCPEGenericESProducer.parameters_()
    genericParams["ComponentName"] = cms.string("PixelCPENNReco")

    process.PixelCPENNReco = cms.ESProducer(
        "PixelCPENNRecoESProducer",
        modelDirectory=cms.string(modelDirectory),
        **genericParams
    )

    if not hasattr(process, "TTRHBuilderAngleAndTemplate"):
        process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi")
    process.TTRHBuilderAngleAndTemplate.PixelCPE = "PixelCPENNReco"
    process.TTRHBuilderAngleAndTemplateWithoutProbQ.PixelCPE = "PixelCPENNReco"
    return process
