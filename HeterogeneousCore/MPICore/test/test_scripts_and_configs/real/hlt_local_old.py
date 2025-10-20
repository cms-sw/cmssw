import FWCore.ParameterSet.Config as cms
import os

# run over HLTPhysics data from run 383363
from hlt import process

process.load("FWCore.MessageLogger.MessageLogger_cfi")

# run with 32 threads, 24 concurrent events, 1 concurrent lumisection, over 10k events
process.options.numberOfThreads = int(os.environ.get("EXPERIMENT_THREADS", 32))
process.options.numberOfStreams = int(os.environ.get("EXPERIMENT_STREAMS", 24))

process.options.numberOfConcurrentLuminosityBlocks = 1  # MPIController does not support concurrent lumisections
process.maxEvents.input = 1000 # 10300

# do not print a final summary
process.options.wantSummary = False
process.MessageLogger.cerr.enableStatistics = cms.untracked.bool(False)


# FastTimer output
experiment_name = os.environ.get("EXPERIMENT_NAME", "unnamed")
output_dir = os.environ.get("EXPERIMENT_OUTPUT_DIR", "../../test_results/one_time_tests/")

process.FastTimerService.writeJSONSummary = True
process.FastTimerService.jsonFileName=cms.untracked.string(f"{output_dir}/local_{experiment_name}.json")

# process.ThroughputService.printEventSummary = True

# set up the MPI communication channel
process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = "file:server.uri"

from HeterogeneousCore.MPICore.mpiController_cfi import mpiController as mpiController_
process.mpiController = mpiController_.clone()

# send the raw data over MPI
process.mpiSenderRawData = cms.EDProducer("MPISender",
    upstream = cms.InputTag("mpiController"),
    instance = cms.int32(1),
    products = cms.vstring("rawDataCollector") 
)

# schedule the communication before the ECAL local reconstruction
process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence.insert(0, process.mpiController)
process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence.insert(1, process.mpiSenderRawData)

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
    ))
)

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
    ))
)

# delete the modules runnig remotely
del process.hltHcalDigisSoA

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
    ))
)

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
    ))
)

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
    ))
)

# schedule the communication before the HBHE local reconstruction
process.HLTDoLocalHcalSequence.insert(0, process.mpiController)
process.HLTDoLocalHcalSequence.insert(1, process.mpiSenderRawData)

process.HLTStoppedHSCPLocalHcalReco.insert(0, process.mpiController)
process.HLTStoppedHSCPLocalHcalReco.insert(1, process.mpiSenderRawData)

# schedule the communication before the HBHE PF reconstruction
process.HLTPFHcalClustering.insert(0, process.mpiController)
process.HLTPFHcalClustering.insert(1, process.mpiSenderRawData)

# schedule the communication for every event
process.Offload = cms.Path(
    process.mpiController +
    process.mpiSenderRawData +
    process.hltHbheRecoSoA +
    process.hltParticleFlowRecHitHBHESoA +
    process.hltParticleFlowClusterHBHESoA +
    process.hltEcalDigisSoA +
    process.hltEcalUncalibRecHitSoA
)

process.schedule.append(process.Offload)

#process.Tracer = cms.Service("Tracer")
