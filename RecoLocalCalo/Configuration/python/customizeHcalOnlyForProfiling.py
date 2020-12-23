import FWCore.ParameterSet.Config as cms

# Customise the HCAL-only reconstruction to run on GPU
#
# Currently, this means:
#   - running the unpacker on CPU, converting the digis into SoA format and copying them to GPU;
#   - running the HBHE local reconstruction, including MAHI, on GPU.
def customizeHcalOnlyForProfilingGPUOnly(process):

  process.consumer = cms.EDAnalyzer("GenericConsumer",
      eventProducts = cms.untracked.vstring('hbheRecHitProducerGPU')
  )

  process.consume_step = cms.EndPath(process.consumer)

  process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.consume_step)

  return process


# Customise the HCAL-only reconstruction to run on GPU, and copy the data to the host
#
# Currently, this means:
#   - running the unpacker on CPU, converting the digis into SoA format and copying them to GPU;
#   - running the HBHE local reconstruction, including MAHI, on GPU;
#   - copying the rechits to CPU and converting them to legacy format.
#
# (this is equivalent to customizeHcalOnlyForProfiling, as the copy and conversion is done by the same module)
def customizeHcalOnlyForProfilingGPUWithHostCopy(process):

  process.consumer = cms.EDAnalyzer("GenericConsumer",
      eventProducts = cms.untracked.vstring('hbheprereco')
  )

  process.consume_step = cms.EndPath(process.consumer)

  process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.consume_step)

  return process


# Customise the HCAL-only reconstruction to run on GPU, copy the data to the host, and convert to legacy format
#
# Currently, this means:
#   - running the unpacker on CPU, converting the digis into SoA format and copying them to GPU;
#   - running the HBHE local reconstruction, including MAHI, on GPU;
#   - copying the rechits to CPU and converting them to legacy format.
#
# The same customisation can be also used on the CPU workflow, running up to the rechits on CPU.
def customizeHcalOnlyForProfiling(process):

  process.consumer = cms.EDAnalyzer("GenericConsumer",
      eventProducts = cms.untracked.vstring('hbheprereco')
  )

  process.consume_step = cms.EndPath(process.consumer)

  process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.consume_step)

  return process
