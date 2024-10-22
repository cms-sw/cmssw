import FWCore.ParameterSet.Config as cms

# Customise the Pixel-only reconstruction to run on GPU
#
# Run the unpacker, clustering, ntuplets, track fit and vertex reconstruction on GPU.
# CUDA and Alpaka co-living here for the moment

def customizePixelOnlyForProfilingGPUOnly(process):

  process.consumer = cms.EDAnalyzer("GenericConsumer",
      eventProducts = cms.untracked.vstring('pixelTracksCUDA', 'pixelVerticesCUDA', '*DeviceProduct_pixelTracksAlpaka_*_*', '*DeviceProduct_pixelVerticesAlpaka_*_*')
  )

  process.consume_step = cms.EndPath(process.consumer)

  process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.consume_step)

  return process


# Customise the Pixel-only reconstruction to run on GPU, and copy the data to the host
#
# Run the unpacker, clustering, ntuplets, track fit and vertex reconstruction on GPU,
# and copy all the products to the host in SoA format.
#
# The same customisation can be also used on the SoA CPU workflow, running up to the
# tracks and vertices on the CPU in SoA format, without conversion to legacy format.
def customizePixelOnlyForProfilingGPUWithHostCopy(process):

  process.consumer = cms.EDAnalyzer("GenericConsumer",
      eventProducts = cms.untracked.vstring('pixelTracksSoA', 'pixelVerticesSoA', 'pixelTracksAlpaka', 'pixelVerticesAlpaka')
  )

  process.consume_step = cms.EndPath(process.consumer)

  process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.consume_step)

  return process


# Customise the Pixel-only reconstruction to run on GPU, copy the data to the host,
# and convert to legacy format
#
# Run the unpacker, clustering, ntuplets, track fit and vertex reconstruction on GPU;
# copy all the products to the host in SoA format; and convert them to legacy format.
#
# The same customisation can be also used on the CPU workflow, running up to the
# tracks and vertices on the CPU.
def customizePixelOnlyForProfiling(process):

  process.consumer = cms.EDAnalyzer("GenericConsumer",
      eventProducts = cms.untracked.vstring('pixelTracks', 'pixelVertices')
  )

  process.consume_step = cms.EndPath(process.consumer)

  process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.consume_step)

  return process
