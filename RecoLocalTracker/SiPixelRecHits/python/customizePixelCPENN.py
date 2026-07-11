import FWCore.ParameterSet.Config as cms

# To set up NN CPE for track refitting, do: 
#from RecoLocalTracker.SiPixelRecHits.customizePixelCPENN import customizePixelCPENN
#process = customizePixelCPENN(process, "<modelDirectory>")


def customizePixelCPENN(process, modelDirectory):
    process.PixelCPENNReco = cms.ESProducer(
        "PixelCPENNRecoESProducer",
        ComponentName=cms.string("PixelCPENNReco"),
        modelDirectory=cms.string(modelDirectory),
    )

    process.TTRHBuilderAngleAndTemplate.PixelCPE = "PixelCPENNReco"
    process.TTRHBuilderAngleAndTemplateWithoutProbQ.PixelCPE = "PixelCPENNReco"
    return process
