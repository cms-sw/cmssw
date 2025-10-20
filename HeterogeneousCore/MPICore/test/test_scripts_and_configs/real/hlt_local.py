import FWCore.ParameterSet.Config as cms
import os

# run over HLTPhysics data from run 383363
from hlt import process

from HLTrigger.Configuration.common import *

process.load("FWCore.MessageLogger.MessageLogger_cfi")

# run with 32 threads, 24 concurrent events, 1 concurrent lumisection, over 10k events
process.options.numberOfThreads = int(os.environ.get("EXPERIMENT_THREADS", 32))
process.options.numberOfStreams = int(os.environ.get("EXPERIMENT_STREAMS", 24))

process.options.numberOfConcurrentLuminosityBlocks = 1  # MPIController does not support concurrent lumisections
process.maxEvents.input = 1000

# do not print a final summary
# process.options.wantSummary = True
process.MessageLogger.cerr.enableStatistics = cms.untracked.bool(False)

# process.writeResults = cms.OutputModule( "PoolOutputModule",
#     fileName = cms.untracked.string( "results2.root" ),
#     compressionAlgorithm = cms.untracked.string( "ZSTD" ),
#     compressionLevel = cms.untracked.int32( 3 ),
#     outputCommands = cms.untracked.vstring( 'keep edmTriggerResults_*_*_*' )
# )

# process.WriteResults = cms.EndPath( process.writeResults )

# process.schedule.append( process.WriteResults )

# Optional: Suppress FwkSummary messages from being printed to cerr or cout
process.MessageLogger.cerr.FwkSummary = cms.untracked.PSet(limit = cms.untracked.int32(0))

# FastTimer output
experiment_name = os.environ.get("EXPERIMENT_NAME", "unnamed")
output_dir = os.environ.get("EXPERIMENT_OUTPUT_DIR", "../../test_results/one_time_tests/")

process.FastTimerService.writeJSONSummary = True
process.FastTimerService.jsonFileName=cms.untracked.string(f"{output_dir}/local_{experiment_name}.json")

process.ThroughputService.printEventSummary = False

# set up the MPI communication channel
process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = "file:server.uri"

from HeterogeneousCore.MPICore.mpiController_cfi import mpiController as mpiController_
process.mpiController = mpiController_.clone()

# process.load("FWCore/Services/Tracer_cfi")

# send the raw data over MPI
process.mpiSenderRawData = cms.EDProducer("MPISender",
    upstream = cms.InputTag("mpiController"),
    instance = cms.int32(1),
    products = cms.vstring("rawDataCollector", "rawDataCollectorActivity")
)


process.hltEcalDigisSoAFilter = cms.EDFilter("PathStateRelease",
    state = cms.InputTag("hltEcalDigisSoA")
    )

insert_modules_before(process, process.hltEcalDigisSoA, process.hltEcalDigisSoAFilter)

del process.hltEcalDigisSoA

# receive the ECAL digis SoA over MPI
process.hltEcalDigisSoA = cms.EDProducer("MPIReceiver",
    upstream = cms.InputTag("mpiSenderRawData"),
    instance = cms.int32(20),
    products = cms.VPSet(cms.PSet(
        type = cms.string("PortableHostCollection<EcalDigiSoALayout<128,false> >"),
        label = cms.string("ebDigis")
    ), cms.PSet(
        type = cms.string("PortableHostCollection<EcalDigiSoALayout<128,false> >"),
        label = cms.string("eeDigis")
    ), cms.PSet(
       type = cms.string("ushort"),
       label = cms.string("backend")
    ),
    cms.PSet(
        type = cms.string("edm::PathStateToken"),
        label = cms.string("")
    ))
)

process.hltEcalUncalibRecHitSoAFilter = cms.EDFilter("PathStateRelease",
    state = cms.InputTag("hltEcalUncalibRecHitSoA")
    )

insert_modules_before(process, process.hltEcalUncalibRecHitSoA, process.hltEcalUncalibRecHitSoAFilter)

del process.hltEcalUncalibRecHitSoA

# receive the ECAL uncalibrated rechits SoA over MPI
process.hltEcalUncalibRecHitSoA = cms.EDProducer("MPIReceiver",
    upstream = cms.InputTag("hltEcalDigisSoA"),
    instance = cms.int32(21),
    products = cms.VPSet(cms.PSet(
        type = cms.string("PortableHostCollection<EcalUncalibratedRecHitSoALayout<128,false> >"),
        label = cms.string("EcalUncalibRecHitsEB")
    ), cms.PSet(
        type = cms.string("PortableHostCollection<EcalUncalibratedRecHitSoALayout<128,false> >"),
        label = cms.string("EcalUncalibRecHitsEE")
    ), cms.PSet(
       type = cms.string("ushort"),
       label = cms.string("backend")
    ),
    cms.PSet(
        type = cms.string("edm::PathStateToken"),
        label = cms.string("")
    ))
)


process.hltHbheRecoSoAFilter = cms.EDFilter("PathStateRelease",
    state = cms.InputTag("hltHbheRecoSoA")
    )

insert_modules_before(process, process.hltHbheRecoSoA, process.hltHbheRecoSoAFilter)

del process.hltHbheRecoSoA

# receive the HBHE rechits SoA over MPI
process.hltHbheRecoSoA = cms.EDProducer("MPIReceiver",
    upstream = cms.InputTag("mpiSenderRawData"),
    instance = cms.int32(11),
    products = cms.VPSet(cms.PSet(
        type = cms.string("PortableHostCollection<hcal::HcalRecHitSoALayout<128,false> >"),
        label = cms.string("")
    ), cms.PSet(
       type = cms.string("ushort"),
       label = cms.string("backend")
    ),
    cms.PSet(
        type = cms.string("edm::PathStateToken"),
        label = cms.string("")
    ))
)

process.hltParticleFlowRecHitHBHESoAFilter = cms.EDFilter("PathStateRelease",
    state = cms.InputTag("hltParticleFlowRecHitHBHESoA")
    )

insert_modules_before(process, process.hltParticleFlowRecHitHBHESoA, process.hltParticleFlowRecHitHBHESoAFilter)

del process.hltParticleFlowRecHitHBHESoA

# receive the HBHE PF rechits SoA over MPI
process.hltParticleFlowRecHitHBHESoA = cms.EDProducer("MPIReceiver",
    upstream = cms.InputTag("hltHbheRecoSoA"),
    instance = cms.int32(12),
    products = cms.VPSet(cms.PSet(
        type = cms.string("PortableHostCollection<reco::PFRecHitSoALayout<128,false> >"),
        label = cms.string("")
    ), cms.PSet(
       type = cms.string("ushort"),
       label = cms.string("backend")
    ),
    cms.PSet(
        type = cms.string("edm::PathStateToken"),
        label = cms.string("")
    ))
)

process.hltParticleFlowClusterHBHESoAFilter = cms.EDFilter("PathStateRelease",
    state = cms.InputTag("hltParticleFlowClusterHBHESoA")
    )

insert_modules_before(process, process.hltParticleFlowClusterHBHESoA, process.hltParticleFlowClusterHBHESoAFilter)

del process.hltParticleFlowClusterHBHESoA

# receive the HBHE PF clusters SoA over MPI
process.hltParticleFlowClusterHBHESoA = cms.EDProducer("MPIReceiver",
    upstream = cms.InputTag("hltParticleFlowRecHitHBHESoA"),
    instance = cms.int32(13),
    products = cms.VPSet(cms.PSet(
        type = cms.string("PortableHostCollection<reco::PFClusterSoALayout<128,false> >"),
        label = cms.string("")
    ), cms.PSet(
        type = cms.string("PortableHostCollection<reco::PFRecHitFractionSoALayout<128,false> >"),
        label = cms.string("")
    ), cms.PSet(
       type = cms.string("ushort"),
       label = cms.string("backend")
    ),
    cms.PSet(
        type = cms.string("edm::PathStateToken"),
        label = cms.string("")
    ))
)

# General path state to validate if the event is active (sometimes it's not apparently)
process.rawDataCollectorActivity = cms.EDProducer("PathStateCapture")

# StateCapture for remote path HLTLocalECAL 
process.EcalDigisAndRecoActivity = cms.EDProducer("PathStateCapture")

# StateCapture for remote path HLTLocalHBHE
process.hltHbheRecoSoAAParticleFlowActivity = cms.EDProducer("PathStateCapture")


# schedule the communication before the ECAL local reconstruction
process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence.insert(0, process.rawDataCollectorActivity) # hltEcalDigisSoA hltEcalUncalibRecHitSoA
process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence.insert(1, process.EcalDigisAndRecoActivity)


# HLTLocalHBHE remote path should run if any of its products will be needed:
# schedule the communication before the HBHE local reconstruction
process.HLTDoLocalHcalSequence.insert(0, process.rawDataCollectorActivity) # hltHbheRecoSoA
process.HLTDoLocalHcalSequence.insert(1, process.hltHbheRecoSoAAParticleFlowActivity)

process.HLTStoppedHSCPLocalHcalReco.insert(0, process.rawDataCollectorActivity) # hltHbheRecoSoA
process.HLTStoppedHSCPLocalHcalReco.insert(1, process.hltHbheRecoSoAAParticleFlowActivity)

# schedule the communication before the HBHE PF reconstruction
process.HLTPFHcalClustering.insert(0, process.rawDataCollectorActivity) # hltParticleFlowRecHitHBHESoA hltParticleFlowClusterHBHESoA
process.HLTPFHcalClustering.insert(1, process.hltHbheRecoSoAAParticleFlowActivity)

process.mpiSenderEcalDigisAndRecoActivity = cms.EDProducer("MPISender",
    upstream = cms.InputTag("mpiController"),
    instance = cms.int32(2),
    products = cms.vstring("EcalDigisAndRecoActivity")
)


process.mpiSenderhltHbheRecoSoAAParticleFlowActivity = cms.EDProducer("MPISender",
    upstream = cms.InputTag("mpiController"),
    instance = cms.int32(3),
    products = cms.vstring("hltHbheRecoSoAAParticleFlowActivity")
)


# schedule the communication for every event
process.Offload = cms.Path(
    process.mpiController +
    process.mpiSenderRawData +
    process.mpiSenderEcalDigisAndRecoActivity +
    process.mpiSenderhltHbheRecoSoAAParticleFlowActivity +
    process.hltHbheRecoSoA +
    process.hltParticleFlowRecHitHBHESoA +
    process.hltParticleFlowClusterHBHESoA +
    process.hltEcalDigisSoA +
    process.hltEcalUncalibRecHitSoA
)

process.schedule.append(process.Offload)

#process.Tracer = cms.Service("Tracer")
