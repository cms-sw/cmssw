import FWCore.ParameterSet.Config as cms

def customizePixelTracksForProfiling(process):
    process.MessageLogger.cerr.FwkReport.reportEvery = 100

    process.out = cms.OutputModule("AsciiOutputModule",
        outputCommands = cms.untracked.vstring(
            "keep *_pixelTracks_*_*",
        ),
        verbosity = cms.untracked.uint32(0),
    )

    process.outPath = cms.EndPath(process.out)

    process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.outPath)

    return process

def customizePixelTracksForProfilingDisableConversion(process):
    process = customizePixelTracksForProfiling(process)

    # Turn off cluster shape filter so that CA doesn't depend on clusters
    process.pixelTracksHitQuadruplets.SeedComparitorPSet = cms.PSet(ComponentName = cms.string("none"))

    # Replace pixel track producer with a dummy one for now
    from RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromCUDA_cfi import pixelTrackProducerFromCUDA as _pixelTrackProducerFromCUDA
    process.pixelTracks = _pixelTrackProducerFromCUDA.clone()

    # Disable conversions to legacy
    process.siPixelClustersPreSplitting.gpuEnableConversion = False
    process.siPixelRecHitsPreSplitting.gpuEnableConversion = False
    process.pixelTracksHitQuadruplets.gpuEnableConversion = False

    return process

def customizePixelTracksForProfilingDisableTransfer(process):
    process = customizePixelTracksForProfilingDisableConversion(process)

    # Disable "unnecessary" transfers to CPU
    process.siPixelClustersPreSplitting.gpuEnableTransfer = False
    process.siPixelRecHitsPreSplitting.gpuEnableTransfer = False
    process.pixelTracksHitQuadruplets.gpuEnableTransfer = False

    return process
