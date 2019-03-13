import FWCore.ParameterSet.Config as cms

def customizePixelTracksForProfiling(process):
    process.MessageLogger.cerr.FwkReport.reportEvery = 100

    process.out = cms.OutputModule("AsciiOutputModule",
        outputCommands = cms.untracked.vstring(
            "keep *_pixelTracks_*_*",
            "keep *_pixelVertices_*_*",
        ),
        verbosity = cms.untracked.uint32(0),
    )

    process.outPath = cms.EndPath(process.out)

    process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.outPath)

    return process

def customizePixelTracksForProfilingDisableConversion(process):
    process = customizePixelTracksForProfiling(process)

    # Disable conversions to legacy
    process.siPixelRecHitsPreSplitting.gpuEnableConversion = False
    process.pixelTracksHitQuadruplets.gpuEnableConversion = False
    process.pixelTracks.gpuEnableConversion = False
    process.pixelVertices.gpuEnableConversion = False

    return process

def customizePixelTracksForProfilingDisableTransfer(process):
    process = customizePixelTracksForProfilingDisableConversion(process)

    # Disable "unnecessary" transfers to CPU
    process.siPixelRecHitsPreSplitting.gpuEnableTransfer = False
    process.pixelTracksHitQuadruplets.gpuEnableTransfer = False
    process.pixelVertices.gpuEnableTransfer = False

    return process
