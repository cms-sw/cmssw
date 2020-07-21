import copy
import FWCore.ParameterSet.Config as cms

# customisation for running on CPUs, common parts
def customise_cpu_common(process):

    # Services

    process.CUDAService = cms.Service("CUDAService",
        enabled = cms.untracked.bool(False)
    )


    # done
    return process


# customisation for offloading to GPUs, common parts
def customise_gpu_common(process):

    # Services

    process.CUDAService = cms.Service("CUDAService",
        enabled = cms.untracked.bool(True),
        allocator = cms.untracked.PSet(
            devicePreallocate = cms.untracked.vuint32(),
        ),
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
    process.hltPixelTracksHitQuadruplets = _caHitNtupletCUDA.clone(
        idealConditions = False,
        pixelRecHitSrc = "hltSiPixelRecHitSoA",
        onGPU = False
    )

    process.hltPixelTracksSoA = cms.EDAlias(
        hltPixelTracksHitQuadruplets = cms.VPSet(
            cms.PSet(
                type = cms.string("32768TrackSoATHeterogeneousSoA")
            )
        )
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
        + process.hltPixelTracksHitQuadruplets              # pixel ntuplets on cpu, in SoA format
        # process.hltPixelTracksSoA                         # alias for hltPixelTracksHitQuadruplets
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
def customise_gpu_ecal(process):

    # FIXME replace the Sequences with empty ones to avoid exanding them during the (re)definition of Modules and EDAliases

    process.HLTDoFullUnpackingEgammaEcalMFSequence = cms.Sequence()
    process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence = cms.Sequence()
    process.HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence()


    # Event Setup

    process.load("EventFilter.EcalRawToDigi.ecalElectronicsMappingGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalGainRatiosGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalPedestalsGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalPulseCovariancesGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalPulseShapesGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalSamplesCorrelationGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalTimeBiasCorrectionsGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalTimeCalibConstantsGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalMultifitParametersGPUESProducer_cfi")

    process.load("RecoLocalCalo.EcalRecProducers.ecalRechitADCToGeVConstantGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalRechitChannelStatusGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalIntercalibConstantsGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalLaserAPDPNRatiosGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalLaserAPDPNRatiosRefGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalLaserAlphasGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalLinearCorrectionsGPUESProducer_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.ecalRecHitParametersGPUESProducer_cfi")


    # Modules and EDAliases

    process.hltEcalDigisGPU = cms.EDProducer("EcalRawToDigiGPU",
        InputLabel = cms.InputTag("rawDataCollector"),
        FEDs = cms.vint32(
            601, 602, 603, 604, 605,
            606, 607, 608, 609, 610,
            611, 612, 613, 614, 615,
            616, 617, 618, 619, 620,
            621, 622, 623, 624, 625,
            626, 627, 628, 629, 630,
            631, 632, 633, 634, 635,
            636, 637, 638, 639, 640,
            641, 642, 643, 644, 645,
            646, 647, 648, 649, 650,
            651, 652, 653, 654
        ),
        digisLabelEB = cms.string("ebDigis"),
        digisLabelEE = cms.string("eeDigis"),
        maxChannelsEB = cms.uint32(61200),
        maxChannelsEE = cms.uint32(14648),
    )

    process.hltEcalDigis = cms.EDProducer("EcalCPUDigisProducer",
        digisInLabelEB = cms.InputTag("hltEcalDigisGPU", "ebDigis"),
        digisInLabelEE = cms.InputTag("hltEcalDigisGPU", "eeDigis"),
        digisOutLabelEB = cms.string("ebDigis"),
        digisOutLabelEE = cms.string("eeDigis"),
        produceDummyIntegrityCollections = cms.bool(True)
    )

    from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitProducerGPU_cfi import ecalUncalibRecHitProducerGPU as _ecalUncalibRecHitProducerGPU
    process.hltEcalUncalibRecHitGPU = _ecalUncalibRecHitProducerGPU.clone(
        digisLabelEB = cms.InputTag("hltEcalDigisGPU", "ebDigis"),
        digisLabelEE = cms.InputTag("hltEcalDigisGPU", "eeDigis")
    )

    process.hltEcalUncalibRecHitSoA = cms.EDProducer("EcalCPUUncalibRecHitProducer",
        containsTimingInformation = cms.bool(False),
        recHitsInLabelEB = cms.InputTag("hltEcalUncalibRecHitGPU", "EcalUncalibRecHitsEB"),
        recHitsInLabelEE = cms.InputTag("hltEcalUncalibRecHitGPU", "EcalUncalibRecHitsEE"),
        recHitsOutLabelEB = cms.string("EcalUncalibRecHitsEB"),
        recHitsOutLabelEE = cms.string("EcalUncalibRecHitsEE")
    )

    process.hltEcalUncalibRecHit = cms.EDProducer("EcalUncalibRecHitConvertGPU2CPUFormat",
        recHitsLabelGPUEB = cms.InputTag("hltEcalUncalibRecHitSoA", "EcalUncalibRecHitsEB"),
        recHitsLabelGPUEE = cms.InputTag("hltEcalUncalibRecHitSoA", "EcalUncalibRecHitsEE"),
        recHitsLabelCPUEB = cms.string("EcalUncalibRecHitsEB"),
        recHitsLabelCPUEE = cms.string("EcalUncalibRecHitsEE")
    )

    # Reconstructing the ECAL calibrated rechits on GPU works, but is extremely slow.
    # Disable it for the time being, until the performance has been addressed.
    """
    process.hltEcalRecHitGPU = cms.EDProducer("EcalRecHitProducerGPU",
        uncalibrecHitsInLabelEB = cms.InputTag("hltEcalUncalibRecHitGPU","EcalUncalibRecHitsEB"),
        uncalibrecHitsInLabelEE = cms.InputTag("hltEcalUncalibRecHitGPU","EcalUncalibRecHitsEE"),
        recHitsLabelEB = cms.string("EcalRecHitsEB"),
        recHitsLabelEE = cms.string("EcalRecHitsEE"),
        maxNumberHitsEB = cms.uint32(61200),
        maxNumberHitsEE = cms.uint32(14648),
        ChannelStatusToBeExcluded = cms.vstring(
            "kDAC",
            "kNoisy",
            "kNNoisy",
            "kFixedG6",
            "kFixedG1",
            "kFixedG0",
            "kNonRespondingIsolated",
            "kDeadVFE",
            "kDeadFE",
            "kNoDataNoTP"),
        killDeadChannels = cms.bool(True),
        EBLaserMIN = cms.double(0.01),
        EELaserMIN = cms.double(0.01),
        EBLaserMAX = cms.double(30.0),
        EELaserMAX = cms.double(30.0),
        flagsMapDBReco = cms.PSet(
            kGood = cms.vstring("kOk","kDAC","kNoLaser","kNoisy"),
            kNoisy = cms.vstring("kNNoisy","kFixedG6","kFixedG1"),
            kNeighboursRecovered = cms.vstring("kFixedG0", "kNonRespondingIsolated", "kDeadVFE"),
            kTowerRecovered = cms.vstring("kDeadFE"),
            kDead = cms.vstring("kNoDataNoTP")
        ),
        recoverEBIsolatedChannels = cms.bool(False),
        recoverEEIsolatedChannels = cms.bool(False),
        recoverEBVFE = cms.bool(False),
        recoverEEVFE = cms.bool(False),
        recoverEBFE = cms.bool(True),
        recoverEEFE = cms.bool(True),
    )

    process.hltEcalRecHitSoA = cms.EDProducer("EcalCPURecHitProducer",
        recHitsInLabelEB = cms.InputTag("hltEcalRecHitGPU", "EcalRecHitsEB"),
        recHitsInLabelEE = cms.InputTag("hltEcalRecHitGPU", "EcalRecHitsEE"),
        recHitsOutLabelEB = cms.string("EcalRecHitsEB"),
        recHitsOutLabelEE = cms.string("EcalRecHitsEE"),
        containsTimingInformation = cms.bool(False),
    )

    process.hltEcalRecHit = cms.EDProducer("EcalRecHitConvertGPU2CPUFormat",
        recHitsLabelGPUEB = cms.InputTag("hltEcalRecHitSoA", "EcalRecHitsEB"),
        recHitsLabelGPUEE = cms.InputTag("hltEcalRecHitSoA", "EcalRecHitsEE"),
        recHitsLabelCPUEB = cms.string("EcalRecHitsEB"),
        recHitsLabelCPUEE = cms.string("EcalRecHitsEE"),
    )
    """


    # Sequences

    process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence = cms.Sequence(
        process.hltEcalDigisGPU                             # unpack ECAL digis on gpu
      + process.hltEcalDigis                                # copy to host and convert to legacy format
      + process.hltEcalUncalibRecHitGPU                     # run ECAL local reconstruction and multifit on gpu
      + process.hltEcalUncalibRecHitSoA                     # needed by hltEcalPhiSymFilter - copy to host
      + process.hltEcalUncalibRecHit                        # needed by hltEcalPhiSymFilter - convert to legacy format
      # process.hltEcalRecHitGPU                            # make ECAL calibrated rechits on gpu
      # process.hltEcalRecHitSoA                            # copy to host
      + process.hltEcalDetIdToBeRecovered                   # legacy producer
      + process.hltEcalRecHit)                              # legacy producer

    process.HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence(
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence
      + process.hltEcalPreshowerDigis                       # unpack ECAL preshower digis on the host
      + process.hltEcalPreshowerRecHit)                     # build ECAL preshower rechits on the host

    process.HLTDoFullUnpackingEgammaEcalMFSequence = copy.copy(process.HLTDoFullUnpackingEgammaEcalSequence)


    # done
    return process

# customisation for offloading the HCAL local reconstruction to GPUs
def customise_gpu_hcal(process):

    # FIXME replace the Sequences with empty ones to avoid exanding them during the (re)definition of Modules and EDAliases

    process.HLTDoLocalHcalSequence = cms.Sequence()
    process.HLTStoppedHSCPLocalHcalReco = cms.Sequence()


    # Event Setup

    process.load("EventFilter.HcalRawToDigi.hcalElectronicsMappingGPUESProducer_cfi")

    process.load("RecoLocalCalo.HcalRecProducers.hcalGainsGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalGainWidthsGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalLUTCorrsGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalConvertedPedestalsGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalConvertedEffectivePedestalsGPUESProducer_cfi")
    process.hcalConvertedEffectivePedestalsGPUESProducer.label0 = "withTopoEff"
    process.load("RecoLocalCalo.HcalRecProducers.hcalConvertedPedestalWidthsGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalConvertedEffectivePedestalWidthsGPUESProducer_cfi")
    process.hcalConvertedEffectivePedestalWidthsGPUESProducer.label0 = "withTopoEff"
    process.hcalConvertedEffectivePedestalWidthsGPUESProducer.label1 = "withTopoEff"
    process.load("RecoLocalCalo.HcalRecProducers.hcalQIECodersGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalRecoParamsWithPulseShapesGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalRespCorrsGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalTimeCorrsGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalQIETypesGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalSiPMParametersGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalSiPMCharacteristicsGPUESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.hcalMahiPulseOffsetsGPUESProducer_cfi")


    # Modules and EDAliases

    # The HCAL unpacker running on the gpu supports only the HB and HE digis.
    # So, run the legacy unacker on the cpu, then convert the HB and HE digis
    # to SoA format and copy them to the gpu.
    process.hltHcalDigisGPU = cms.EDProducer("HcalDigisProducerGPU",
        hbheDigisLabel = cms.InputTag("hltHcalDigis"),
        qie11DigiLabel = cms.InputTag("hltHcalDigis"),
        digisLabelF01HE = cms.string(""),
        digisLabelF5HB = cms.string(""),
        digisLabelF3HB = cms.string(""),
        maxChannelsF01HE = cms.uint32(10000),
        maxChannelsF5HB = cms.uint32(10000),
        maxChannelsF3HB = cms.uint32(10000),
        nsamplesF01HE = cms.uint32(8),
        nsamplesF5HB = cms.uint32(8),
        nsamplesF3HB = cms.uint32(8)
    )

    from RecoLocalCalo.HcalRecProducers.hbheRecHitProducerGPU_cfi import hbheRecHitProducerGPU as _hbheRecHitProducerGPU
    process.hltHbherecoGPU = _hbheRecHitProducerGPU.clone(
        digisLabelF01HE = "hltHcalDigisGPU",
        digisLabelF5HB = "hltHcalDigisGPU",
        digisLabelF3HB = "hltHcalDigisGPU",
        recHitsLabelM0HBHE = ""
    )

    from RecoLocalCalo.HcalRecProducers.hcalCPURecHitsProducer_cfi import hcalCPURecHitsProducer as _hcalCPURecHitsProducer
    process.hltHbhereco = _hcalCPURecHitsProducer.clone(
        recHitsM0LabelIn = "hltHbherecoGPU",
        recHitsM0LabelOut = "",
        recHitsLegacyLabelOut = ""
    )


    # Sequences

    process.HLTDoLocalHcalSequence = cms.Sequence(
          process.hltHcalDigis                              # legacy producer, unpack HCAL digis on cpu
        + process.hltHcalDigisGPU                           # copy to gpu and convert to SoA format
        + process.hltHbherecoGPU                            # run HCAL local reconstruction (Method 0 and MAHI) on gpu
        + process.hltHbhereco                               # copy to host and convert to legacy format
        + process.hltHfprereco                              # legacy producer
        + process.hltHfreco                                 # legacy producer
        + process.hltHoreco)                                # legacy producer

    process.HLTStoppedHSCPLocalHcalReco = cms.Sequence(
          process.hltHcalDigis                              # legacy producer, unpack HCAL digis on cpu
        + process.hltHcalDigisGPU                           # copy to gpu and convert to SoA format
        + process.hltHbherecoGPU                            # run HCAL local reconstruction (Method 0 and MAHI) on gpu
        + process.hltHbhereco)                              # copy to host and convert to legacy format


    # done
    return process


# customisation for running on CPUs
def customise_for_Patatrack_on_cpu(process):
    process = customise_cpu_common(process)
    process = customise_cpu_pixel(process)
    return process


# customisation for offloading to GPUs
def customise_for_Patatrack_on_gpu(process):
    process = customise_gpu_common(process)
    process = customise_gpu_pixel(process)
    process = customise_gpu_ecal(process)
    process = customise_gpu_hcal(process)
    return process
