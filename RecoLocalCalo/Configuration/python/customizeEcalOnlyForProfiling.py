import FWCore.ParameterSet.Config as cms

# Customise the ECAL-only reconstruction to run on GPU
#
# Currently, this means running only the unpacker and multifit, up to the uncalbrated rechits
def customizeEcalOnlyForProfilingGPUOnly(process):

  process.consumer = cms.EDAnalyzer("GenericConsumer",
      eventProducts = cms.untracked.vstring('ecalMultiFitUncalibRecHitGPU')
  )

  process.consume_step = cms.EndPath(process.consumer)

  process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.consume_step)

  return process


# Customise the ECAL-only reconstruction to run on GPU, and copy the data to the host
#
# Currently, this means running only the unpacker and multifit, up to the uncalbrated rechits
def customizeEcalOnlyForProfilingGPUWithHostCopy(process):

  process.consumer = cms.EDAnalyzer("GenericConsumer",
      eventProducts = cms.untracked.vstring('ecalMultiFitUncalibRecHitSoA')
  )

  process.consume_step = cms.EndPath(process.consumer)

  process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.consume_step)

  return process


# Customise the ECAL-only reconstruction to run on GPU, copy the data to the host, and convert to legacy format
#
# Currently, this means running only the unpacker and multifit, up to the uncalbrated rechits, on the GPU
# and the rechits producer on the CPU
#
# The same customisation can be also used on the CPU workflow, running up to the rechits on CPU.
def customizeEcalOnlyForProfiling(process):

  process.consumer = cms.EDAnalyzer("GenericConsumer",
      eventProducts = cms.untracked.vstring('ecalRecHit')
  )

  process.consume_step = cms.EndPath(process.consumer)

  process.schedule = cms.Schedule(process.raw2digi_step, process.reconstruction_step, process.consume_step)

  return process
