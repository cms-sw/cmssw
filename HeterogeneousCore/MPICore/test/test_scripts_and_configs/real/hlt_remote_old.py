import FWCore.ParameterSet.Config as cms

import os

from hlt import process as _process

process = cms.Process("REMOTE")

process.load("Configuration.StandardSequences.Accelerators_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# load the event setup
for module in _process.psets.keys():
    setattr(process, module, getattr(_process, module).clone())
for module in _process.es_sources.keys():
    setattr(process, module, getattr(_process, module).clone())
for module in _process.es_producers.keys():
    setattr(process, module, getattr(_process, module).clone())


process.options.numberOfThreads = int(os.environ.get("EXPERIMENT_THREADS", 32))
process.options.numberOfStreams = int(os.environ.get("EXPERIMENT_STREAMS", 24))
process.options.numberOfConcurrentLuminosityBlocks = 1


# FastTimer output
experiment_name = os.environ.get("EXPERIMENT_NAME", "unnamed")
output_dir = os.environ.get("EXPERIMENT_OUTPUT_DIR", "../../test_results/one_time_tests/")


process.FastTimerService = _process.FastTimerService.clone()
process.FastTimerService.writeJSONSummary = True
process.FastTimerService.jsonFileName=cms.untracked.string(f"{output_dir}/remote_{experiment_name}.json")

process.ThroughputService = _process.ThroughputService.clone()
# process.ThroughputService.printEventSummary = True

# set up the MPI communication channel
process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = "file:server.uri"

process.source = cms.Source("MPISource",
  firstRun = cms.untracked.uint32(383631)
)

process.maxEvents.input = -1

# receive the raw data over MPI
process.rawDataCollector = cms.EDProducer("MPIReceiver",
    upstream = cms.InputTag("source"),
    instance = cms.int32(1),
    products = cms.VPSet(cms.PSet(
        type = cms.string("FEDRawDataCollection"),
        label = cms.string("")
    ))
)

process.hltGetRaw = _process.hltGetRaw.clone()

# HBHE local reconstruction from the HLT menu
process.hltHcalDigis = _process.hltHcalDigis.clone()
process.hltHcalDigisSoA = _process.hltHcalDigisSoA.clone()
process.hltHbheRecoSoA = _process.hltHbheRecoSoA.clone()
process.hltParticleFlowRecHitHBHESoA = _process.hltParticleFlowRecHitHBHESoA.clone()
process.hltParticleFlowClusterHBHESoA = _process.hltParticleFlowClusterHBHESoA.clone()

# send the HBHE rechits SoA over MPI
process.mpiSenderHbheRecoSoA = cms.EDProducer("MPISender",
    upstream = cms.InputTag("rawDataCollector"),
    instance = cms.int32(11),
    products = cms.vstring(
        "128falsehcalHcalRecHitSoALayoutPortableHostCollection_hltHbheRecoSoA__*",
        "ushort_hltHbheRecoSoA_backend_*",
    ) 
)

# send the HBHE PF rechits SoA over MPI
process.mpiSenderParticleFlowRecHitHBHESoA = cms.EDProducer("MPISender",
    upstream = cms.InputTag("mpiSenderHbheRecoSoA"),
    instance = cms.int32(12),
    products = cms.vstring(
        "128falserecoPFRecHitSoALayoutPortableHostCollection_hltParticleFlowRecHitHBHESoA__*",
        "ushort_hltParticleFlowRecHitHBHESoA_backend_*",
    )
)

# send the HBHE PF clusters SoA over MPI
process.mpiSenderParticleFlowClusterHBHESoA = cms.EDProducer("MPISender",
    upstream = cms.InputTag("mpiSenderParticleFlowRecHitHBHESoA"),
    instance = cms.int32(13),
    products = cms.vstring(
        "128falserecoPFClusterSoALayoutPortableHostCollection_hltParticleFlowClusterHBHESoA__*",
        "128falserecoPFRecHitFractionSoALayoutPortableHostCollection_hltParticleFlowClusterHBHESoA__*",
        "ushort_hltParticleFlowClusterHBHESoA_backend_*",
    )
)

# run the HBHE local reconstruction
process.HLTLocalHBHE = cms.Path(
    process.rawDataCollector +
    process.hltGetRaw +
    process.hltHcalDigis +
    process.hltHcalDigisSoA +
    process.hltHbheRecoSoA +
    process.mpiSenderHbheRecoSoA +
    process.hltParticleFlowRecHitHBHESoA +
    process.mpiSenderParticleFlowRecHitHBHESoA +
    process.hltParticleFlowClusterHBHESoA +
    process.mpiSenderParticleFlowClusterHBHESoA
)

# ECAL local reconstruction from the HLT menu
process.hltEcalDigisSoA = _process.hltEcalDigisSoA.clone()
process.hltEcalUncalibRecHitSoA = _process.hltEcalUncalibRecHitSoA.clone()

# send the ECAL digis SoA over MPI
process.mpiSenderEcalDigisSoA = cms.EDProducer("MPISender",
    upstream = cms.InputTag("rawDataCollector"),
    instance = cms.int32(20),
    products = cms.vstring(
        "128falseEcalDigiSoALayoutPortableHostCollection_hltEcalDigisSoA_ebDigis_*",
        "128falseEcalDigiSoALayoutPortableHostCollection_hltEcalDigisSoA_eeDigis_*",
        "ushort_hltEcalDigisSoA_backend_*",
    ) 
)

# send the ECAL uncalibrated rechits SoA over MPI
process.mpiSenderEcalUncalibRecHitSoA = cms.EDProducer("MPISender",
    upstream = cms.InputTag("mpiSenderEcalDigisSoA"),
    instance = cms.int32(21),
    products = cms.vstring(
        "128falseEcalUncalibratedRecHitSoALayoutPortableHostCollection_hltEcalUncalibRecHitSoA_EcalUncalibRecHitsEB_*",
        "128falseEcalUncalibratedRecHitSoALayoutPortableHostCollection_hltEcalUncalibRecHitSoA_EcalUncalibRecHitsEE_*",
        "ushort_hltEcalUncalibRecHitSoA_backend_*",
    ) 
)

# run the ECAL local reconstruction
process.HLTLocalECAL = cms.Path(
    process.rawDataCollector +
    process.hltGetRaw +
    process.hltEcalDigisSoA +
    process.mpiSenderEcalDigisSoA +
    process.hltEcalUncalibRecHitSoA +
    process.mpiSenderEcalUncalibRecHitSoA
)

# schedule the reconstruction
process.schedule = cms.Schedule(
    process.HLTLocalHBHE,
    process.HLTLocalECAL
)

#process.Tracer = cms.Service("Tracer")
