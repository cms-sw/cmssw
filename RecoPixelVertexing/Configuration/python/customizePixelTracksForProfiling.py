import FWCore.ParameterSet.Config as cms

def customizePixelTracksForProfilingGPUOnly(process):
    process.MessageLogger.cerr.FwkReport.reportEvery = 100

    process.Raw2Hit = cms.Path(process.offlineBeamSpot+process.offlineBeamSpotCUDA+process.siPixelClustersCUDAPreSplitting+process.siPixelRecHitsCUDAPreSplitting)

    process.load('RecoPixelVertexing.PixelTriplets.caHitNtupletCUDA_cfi')
    process.load('RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi')
    process.TVreco = cms.Path(process.caHitNtupletCUDA+process.pixelVertexCUDA)

    process.schedule = cms.Schedule(process.Raw2Hit, process.TVreco)
    return process

def customizePixelTracksForProfilingSoAonCPU(process):
    process = customizePixelTracksForProfilingGPUOnly(process)

    process.pixelVertexSoA = process.pixelVertexCUDA.clone()
    process.pixelVertexSoA.onGPU = False
    process.pixelVertexSoA.pixelTrackSrc = 'pixelTrackSoA'
    process.TVSoAreco = cms.Path(process.caHitNtupletCUDA+process.pixelTrackSoA+process.pixelVertexSoA)

    process.schedule = cms.Schedule(process.Raw2Hit, process.TVSoAreco)

    return process

def customizePixelTracksForProfilingEnableTransfer(process):
    process = customizePixelTracksForProfilingGPUOnly(process)

    process.load('RecoPixelVertexing.PixelTrackFitting.pixelTrackSoA_cfi')
    process.load('RecoPixelVertexing.PixelVertexFinding.pixelVertexSoA_cfi')
    process.toSoA = cms.Path(process.pixelTrackSoA+process.pixelVertexSoA)

    process.schedule = cms.Schedule(process.Raw2Hit, process.TVreco, process.toSoA)
    return process

def customizePixelTracksForProfilingEnableConversion(process):
    # use old trick of output path
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

