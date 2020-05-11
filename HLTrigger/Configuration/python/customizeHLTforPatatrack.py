import FWCore.ParameterSet.Config as cms

# customisation for offloading to GPUs, common parts
def customise_gpu_common(process):

    # Services

    process.CUDAService = cms.Service("CUDAService",
        allocator = cms.untracked.PSet(
            devicePreallocate = cms.untracked.vuint32(),
        ),
        enabled = cms.untracked.bool(True),
        limits = cms.untracked.PSet(
            cudaLimitDevRuntimePendingLaunchCount = cms.untracked.int32(-1),
            cudaLimitDevRuntimeSyncDepth = cms.untracked.int32(-1),
            cudaLimitMallocHeapSize = cms.untracked.int32(-1),
            cudaLimitPrintfFifoSize = cms.untracked.int32(-1),
            cudaLimitStackSize = cms.untracked.int32(-1)
        )
    )

    process.load("HeterogeneousCore.CUDAServices.NVProfilerService_cfi")

    # done
    return process


# customisation for running the "Patatrack" pixel track reconstruction on CPUs
def customise_cpu_pixel(process):

    # FIXME replace the Sequences with empty ones to avoid exanding them during the (re)definition of Modules and EDAliases

    process.HLTRecoPixelTracksSequence = cms.Sequence()
    process.HLTRecopixelvertexingSequence = cms.Sequence()


    # Event Setup

    process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEFastESProducer_cfi")
    process.PixelCPEFastESProducer.DoLorentz = True


    # Modules and EDAliases

    # referenced in process.HLTRecoPixelTracksSequence

    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitHostSoA_cfi import siPixelRecHitHostSoA as _siPixelRecHitHostSoA
    process.hltSiPixelRecHitSoA = _siPixelRecHitHostSoA.clone(
        beamSpot = "hltOnlineBeamSpot",
        src = "hltSiPixelClusters",
        convertToLegacy = True
    )

    from RecoPixelVertexing.PixelTriplets.caHitNtupletCUDA_cfi import caHitNtupletCUDA as _caHitNtupletCUDA
    process.hltPixelTracksSoA = _caHitNtupletCUDA.clone(
        idealConditions = False,
        pixelRecHitSrc = "hltSiPixelRecHitSoA",
        onGPU = False
    )

    from RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi import pixelTrackProducerFromSoA as _pixelTrackProducerFromSoA
    process.hltPixelTracks = _pixelTrackProducerFromSoA.clone(
        beamSpot = "hltOnlineBeamSpot",
        pixelRecHitLegacySrc = "hltSiPixelRecHits",
        trackSrc = "hltPixelTracksSoA"
    )


    # referenced in process.HLTRecopixelvertexingSequence

    from RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi import pixelVertexCUDA as _pixelVertexCUDA
    process.hltPixelVerticesSoA = _pixelVertexCUDA.clone(
        pixelTrackSrc = "hltPixelTracksSoA",
        onGPU = False
    )

    from RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi import pixelVertexFromSoA as _pixelVertexFromSoA
    process.hltPixelVertices = _pixelVertexFromSoA.clone(
        TrackCollection = "hltPixelTracks",
        beamSpot = "hltOnlineBeamSpot",
        src = "hltPixelVerticesSoA"
    )


    # Sequences

    process.HLTRecoPixelTracksSequence = cms.Sequence(
          process.hltPixelTracksFitter                      # not used here, kept for compatibility with legacy sequences
        + process.hltPixelTracksFilter                      # not used here, kept for compatibility with legacy sequences
        + process.hltPixelTracksTrackingRegions             # from the original sequence
        + process.hltSiPixelRecHitSoA                       # pixel rechits on cpu, converted to SoA
        + process.hltPixelTracksSoA                         # pixel ntuplets on cpu, in SoA format
        + process.hltPixelTracks)                           # pixel tracks on cpu, with conversion to legacy

    process.HLTRecopixelvertexingSequence = cms.Sequence(
         process.HLTRecoPixelTracksSequence
       + process.hltPixelVerticesSoA                        # pixel vertices on cpu, in SoA format
       + process.hltPixelVertices                           # pixel vertices on cpu, in legacy format
       + process.hltTrimmedPixelVertices)                   # from the original sequence


    # done
    return process


# customisation for offloading the Pixel local reconstruction to GPUs
def customise_gpu_pixel(process):

    # FIXME replace the Sequences with empty ones to avoid exanding them during the (re)definition of Modules and EDAliases

    process.HLTDoLocalPixelSequence = cms.Sequence()
    process.HLTRecoPixelTracksSequence = cms.Sequence()
    process.HLTRecopixelvertexingSequence = cms.Sequence()


    # Event Setup

    process.load("CalibTracker.SiPixelESProducers.siPixelGainCalibrationForHLTGPU_cfi")
    process.load("RecoLocalTracker.SiPixelClusterizer.siPixelFedCablingMapGPUWrapper_cfi")
    process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEFastESProducer_cfi")
    process.PixelCPEFastESProducer.DoLorentz = True


    # Modules and EDAliases

    # referenced in process.HLTDoLocalPixelSequence

    process.hltOnlineBeamSpotCUDA = cms.EDProducer("BeamSpotToCUDA",
        src = cms.InputTag("hltOnlineBeamSpot")
    )

    from RecoLocalTracker.SiPixelClusterizer.siPixelRawToClusterCUDA_cfi import siPixelRawToClusterCUDA as _siPixelRawToClusterCUDA
    process.hltSiPixelClustersCUDA = _siPixelRawToClusterCUDA.clone()

    process.hltSiPixelRecHitsCUDA = cms.EDProducer("SiPixelRecHitCUDA",
        CPE = cms.string("PixelCPEFast"),
        beamSpot = cms.InputTag("hltOnlineBeamSpotCUDA"),
        src = cms.InputTag("hltSiPixelClustersCUDA")
    )

    process.hltSiPixelDigisSoA = cms.EDProducer("SiPixelDigisSoAFromCUDA",
        src = cms.InputTag("hltSiPixelClustersCUDA")
    )

    process.hltSiPixelDigisClusters = cms.EDProducer("SiPixelDigisClustersFromSoA",
        src = cms.InputTag("hltSiPixelDigisSoA")
    )

    process.hltSiPixelDigiErrorsSoA = cms.EDProducer("SiPixelDigiErrorsSoAFromCUDA",
        src = cms.InputTag("hltSiPixelClustersCUDA")
    )

    from EventFilter.SiPixelRawToDigi.siPixelDigiErrorsFromSoA_cfi import siPixelDigiErrorsFromSoA as _siPixelDigiErrorsFromSoA
    process.hltSiPixelDigiErrors = _siPixelDigiErrorsFromSoA.clone(
        UsePhase1 = True,
        digiErrorSoASrc = "hltSiPixelDigiErrorsSoA"
    )

    process.hltSiPixelRecHits = cms.EDProducer("SiPixelRecHitFromSOA",
        pixelRecHitSrc = cms.InputTag("hltSiPixelRecHitsCUDA"),
        src = cms.InputTag("hltSiPixelDigisClusters")
    )

    process.hltSiPixelDigis = cms.EDAlias(
        hltSiPixelDigisClusters = cms.VPSet(
            cms.PSet(
                type = cms.string("PixelDigiedmDetSetVector")
            )
        ),
        hltSiPixelDigiErrors = cms.VPSet(
            cms.PSet(
                type = cms.string("DetIdedmEDCollection")
            ),
            cms.PSet(
                type = cms.string("SiPixelRawDataErroredmDetSetVector")
            ),
            cms.PSet(
                type = cms.string("PixelFEDChanneledmNewDetSetVector")
            )
        )
    )

    process.hltSiPixelClusters = cms.EDAlias(
        hltSiPixelDigisClusters = cms.VPSet(
            cms.PSet(
                type = cms.string("SiPixelClusteredmNewDetSetVector")
            )
        )
    )

    # referenced in process.HLTRecoPixelTracksSequence

    from RecoPixelVertexing.PixelTriplets.caHitNtupletCUDA_cfi import caHitNtupletCUDA as _caHitNtupletCUDA
    process.hltPixelTracksHitQuadruplets = _caHitNtupletCUDA.clone(
        idealConditions = False,
        pixelRecHitSrc = "hltSiPixelRecHitsCUDA",
        onGPU = True
    )

    process.hltPixelTracksSoA = cms.EDProducer("PixelTrackSoAFromCUDA",
        src = cms.InputTag("hltPixelTracksHitQuadruplets")
    )

    process.hltPixelTracks = cms.EDProducer("PixelTrackProducerFromSoA",
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        minNumberOfHits = cms.int32(0),
        pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
        trackSrc = cms.InputTag("hltPixelTracksSoA")
    )

    # referenced in process.HLTRecopixelvertexingSequence

    from RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi import pixelVertexCUDA as _pixelVertexCUDA
    process.hltPixelVerticesCUDA = _pixelVertexCUDA.clone(
        pixelTrackSrc = "hltPixelTracksHitQuadruplets",
        onGPU = True
    )

    process.hltPixelVerticesSoA = cms.EDProducer("PixelVertexSoAFromCUDA",
        src = cms.InputTag("hltPixelVerticesCUDA")
    )

    process.hltPixelVertices = cms.EDProducer("PixelVertexProducerFromSoA",
        src = cms.InputTag("hltPixelVerticesSoA"),
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        TrackCollection = cms.InputTag("hltPixelTracks"),
    )


    # Sequences

    process.HLTDoLocalPixelSequence = cms.Sequence(
          process.hltOnlineBeamSpotCUDA                     # transfer the beamspot to the gpu
        + process.hltSiPixelClustersCUDA                    # digis and clusters on gpu
        + process.hltSiPixelRecHitsCUDA                     # rechits on gpu
        + process.hltSiPixelDigisSoA                        # copy to host
        + process.hltSiPixelDigisClusters                   # convert to legacy
        + process.hltSiPixelDigiErrorsSoA                   # copy to host
        + process.hltSiPixelDigiErrors                      # convert to legacy
        # process.hltSiPixelDigis                           # replaced by an alias
        # process.hltSiPixelClusters                        # replaced by an alias
        + process.hltSiPixelClustersCache                   # not used here, kept for compatibility with legacy sequences
        + process.hltSiPixelRecHits)                        # convert to legacy

    process.HLTRecoPixelTracksSequence = cms.Sequence(
          process.hltPixelTracksFitter                      # not used here, kept for compatibility with legacy sequences
        + process.hltPixelTracksFilter                      # not used here, kept for compatibility with legacy sequences
        + process.hltPixelTracksTrackingRegions             # from the original sequence
        + process.hltPixelTracksHitQuadruplets              # pixel ntuplets on gpu, in SoA format
        + process.hltPixelTracksSoA                         # pixel ntuplets on cpu, in SoA format
        + process.hltPixelTracks)                           # pixel tracks on gpu, with transfer and conversion to legacy

    process.HLTRecopixelvertexingSequence = cms.Sequence(
         process.HLTRecoPixelTracksSequence
       + process.hltPixelVerticesCUDA                       # pixel vertices on gpu, in SoA format
       + process.hltPixelVerticesSoA                        # pixel vertices on cpu, in SoA format
       + process.hltPixelVertices                           # pixel vertices on cpu, in legacy format
       + process.hltTrimmedPixelVertices)                   # from the original sequence


    # done

    return process


# customisation for offloading the ECAL local reconstruction to GPUs
# TODO find automatically the list of Sequences to be updated
def customise_gpu_ecal(process):

    # FIXME replace the Sequences with empty ones to avoid exanding them during the (re)definition of Modules and EDAliases

    process.HLTDoFullUnpackingEgammaEcalMFSequence = cms.Sequence()
    process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence = cms.Sequence()
    process.HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence()


    # Event Setup

    process.load("RecoLocalCalo.EcalRecProducers.ecalGainRatiosGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalPedestalsGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalPulseCovariancesGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalPulseShapesGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalSamplesCorrelationGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalTimeBiasCorrectionsGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalTimeCalibConstantsGPUESProducer_cfi")


    # Modules and EDAliases

    process.hltEcalUncalibRecHitSoA = cms.EDProducer("EcalUncalibRecHitProducerGPU",
        digisLabelEB = cms.InputTag("hltEcalDigis","ebDigis"),
        recHitsLabelEB = cms.string("EcalUncalibRecHitsEB"),
        digisLabelEE = cms.InputTag("hltEcalDigis","eeDigis"),
        recHitsLabelEE = cms.string("EcalUncalibRecHitsEE"),
        EBamplitudeFitParameters = cms.vdouble(1.138, 1.652),
        EBtimeConstantTerm = cms.double(0.6),
        EBtimeFitLimits_Lower = cms.double(0.2),
        EBtimeFitLimits_Upper = cms.double(1.4),
        EBtimeFitParameters = cms.vdouble(-2.015452, 3.130702, -12.3473, 41.88921, -82.83944, 91.01147, -50.35761, 11.05621),
        EBtimeNconst = cms.double(28.5),
        EEamplitudeFitParameters = cms.vdouble(1.89, 1.4),
        EEtimeConstantTerm = cms.double(1.0),
        EEtimeFitLimits_Lower = cms.double(0.2),
        EEtimeFitLimits_Upper = cms.double(1.4),
        EEtimeFitParameters = cms.vdouble(-2.390548, 3.553628, -17.62341, 67.67538, -133.213, 140.7432, -75.41106, 16.20277),
        EEtimeNconst = cms.double(31.8),
        amplitudeThresholdEB = cms.double(10.0),
        amplitudeThresholdEE = cms.double(10.0),
        outOfTimeThresholdGain12mEB = cms.double(5.0),
        outOfTimeThresholdGain12mEE = cms.double(1000.0),
        outOfTimeThresholdGain12pEB = cms.double(5.0),
        outOfTimeThresholdGain12pEE = cms.double(1000.0),
        outOfTimeThresholdGain61mEB = cms.double(5.0),
        outOfTimeThresholdGain61mEE = cms.double(1000.0),
        outOfTimeThresholdGain61pEB = cms.double(5.0),
        outOfTimeThresholdGain61pEE = cms.double(1000.0),
        kernelMinimizeThreads = cms.vuint32(32, 1, 1),
        maxNumberHits = cms.uint32(20000),
        shouldRunTimingComputation = cms.bool(False),
        shouldTransferToHost = cms.bool(True)
    )

    process.hltEcalUncalibRecHit = cms.EDProducer("EcalUncalibRecHitConvertGPU2CPUFormat",
        recHitsLabelGPUEB = cms.InputTag("hltEcalUncalibRecHitSoA", "EcalUncalibRecHitsEB"),
        recHitsLabelGPUEE = cms.InputTag("hltEcalUncalibRecHitSoA", "EcalUncalibRecHitsEE"),
        recHitsLabelCPUEB = cms.string("EcalUncalibRecHitsEB"),
        recHitsLabelCPUEE = cms.string("EcalUncalibRecHitsEE")
    )


    # Sequences

    process.HLTDoFullUnpackingEgammaEcalMFSequence = cms.Sequence(
        process.hltEcalDigis
      + process.hltEcalPreshowerDigis
      + process.hltEcalUncalibRecHitSoA
      + process.hltEcalUncalibRecHit
      + process.hltEcalDetIdToBeRecovered
      + process.hltEcalRecHit
      + process.hltEcalPreshowerRecHit)

    process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence = cms.Sequence(
        process.hltEcalDigis
      + process.hltEcalUncalibRecHitSoA
      + process.hltEcalUncalibRecHit
      + process.hltEcalDetIdToBeRecovered
      + process.hltEcalRecHit)

    process.HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence(
        process.hltEcalDigis
      + process.hltEcalPreshowerDigis
      + process.hltEcalUncalibRecHitSoA
      + process.hltEcalUncalibRecHit
      + process.hltEcalDetIdToBeRecovered
      + process.hltEcalRecHit
      + process.hltEcalPreshowerRecHit)


    # done
    return process


# customisation for running on CPUs
def customise_for_Patatrack_on_cpu(process):
    process = customise_cpu_pixel(process)
    return process


# customisation for offloading to GPUs
def customise_for_Patatrack_on_gpu(process):
    process = customise_gpu_common(process)
    process = customise_gpu_pixel(process)
    #process = customise_gpu_ecal(process)
    return process

