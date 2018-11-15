import FWCore.ParameterSet.Config as cms

def customizePixelTracksForProfiling(process):
    process.out = cms.OutputModule("AsciiOutputModule",
        outputCommands = cms.untracked.vstring(
            "keep *_pixelTracks_*_*",
        ),
        verbosity = cms.untracked.uint32(0),
    )

    process.outPath = cms.EndPath(process.out)

    process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.outPath)

    return process
