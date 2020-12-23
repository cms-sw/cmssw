import copy
import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
from HLTrigger.Configuration.common import *
from Configuration.Eras.Modifier_run3_common_cff import run3_common


# force the SwitchProducerCUDA choice to pick a specific backend: True for offloading to a gpu, False for running on cpu
def forceGpuOffload(status = True):
    import HeterogeneousCore.CUDACore.SwitchProducerCUDA
    HeterogeneousCore.CUDACore.SwitchProducerCUDA._cuda_enabled_cached = bool(status)


# reset the SwitchProducerCUDA choice to pick a backend depending on the availability of a supported gpu
def resetGpuOffload():
    import HeterogeneousCore.CUDACore.SwitchProducerCUDA
    HeterogeneousCore.CUDACore.SwitchProducerCUDA._cuda_enabled_cached = None
    HeterogeneousCore.CUDACore.SwitchProducerCUDA._switch_cuda()


# check if CUDA is enabled, using the same mechanism as the SwitchProducerCUDA
def cudaIsEnabled():
    import HeterogeneousCore.CUDACore.SwitchProducerCUDA
    return HeterogeneousCore.CUDACore.SwitchProducerCUDA._switch_cuda()[0]


# customisation for running the Patatrack reconstruction, common parts
def customiseCommon(process):

    # Services

    process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
    process.CUDAService.enabled = cudaIsEnabled()

    process.load("HeterogeneousCore.CUDAServices.NVProfilerService_cfi")


    # Paths and EndPaths

    # the hltGetConditions module would force gpu-specific ESProducers to run even if no supported gpu is present
    if 'hltGetConditions' in process.__dict__:
        del process.hltGetConditions

    # produce a boolean to track if the events ar being processed on gpu (true) or cpu (false)
    process.statusOnGPU = SwitchProducerCUDA(
        cpu  = cms.EDProducer("BooleanProducer", value = cms.bool(False)),
        cuda = cms.EDProducer("BooleanProducer", value = cms.bool(True))
    )

    process.statusOnGPUFilter = cms.EDFilter("BooleanFilter",
        src = cms.InputTag("statusOnGPU")
    )

    if 'Status_OnGPU' in process.__dict__:
        replace_with(process.Status_OnGPU, cms.Path(process.statusOnGPU + process.statusOnGPUFilter))
    else:
        process.Status_OnGPU = cms.Path(process.statusOnGPU + process.statusOnGPUFilter)
        if 'HLTSchedule' in process.__dict__:
            process.HLTSchedule.append(process.Status_OnGPU)
        if process.schedule is not None:
            process.schedule.append(process.Status_OnGPU)


    # make the ScoutingCaloMuonOutput endpath compatible with using Tasks in the Scouting paths
    if 'hltOutputScoutingCaloMuon' in process.__dict__ and not 'hltPreScoutingCaloMuonOutputSmart' in process.__dict__:
        process.hltPreScoutingCaloMuonOutputSmart = cms.EDFilter( "TriggerResultsFilter",
            l1tIgnoreMaskAndPrescale = cms.bool( False ),
            l1tResults = cms.InputTag( "" ),
            hltResults = cms.InputTag( 'TriggerResults','','@currentProcess' ),
            triggerConditions = process.hltOutputScoutingCaloMuon.SelectEvents.SelectEvents,
            throw = cms.bool( True )
        )
        insert_modules_after(process, process.hltPreScoutingCaloMuonOutput, process.hltPreScoutingCaloMuonOutputSmart)

    # make the ScoutingPFOutput endpath compatible with using Tasks in the Scouting paths
    if 'hltOutputScoutingPF' in process.__dict__ and not 'hltPreScoutingPFOutputSmart' in process.__dict__:
        process.hltPreScoutingPFOutputSmart = cms.EDFilter( "TriggerResultsFilter",
            l1tIgnoreMaskAndPrescale = cms.bool( False ),
            l1tResults = cms.InputTag( "" ),
            hltResults = cms.InputTag( 'TriggerResults','','@currentProcess' ),
            triggerConditions = process.hltOutputScoutingPF.SelectEvents.SelectEvents,
            throw = cms.bool( True )
        )
        insert_modules_after(process, process.hltPreScoutingPFOutput, process.hltPreScoutingPFOutputSmart)


    # done
    return process


# customisation for running the "Patatrack" pixel local reconstruction
def customisePixelLocalReconstruction(process):

    if not 'HLTDoLocalPixelSequence' in process.__dict__:
        return process


    # FIXME replace the Sequences with empty ones to avoid exanding them during the (re)definition of Modules and EDAliases

    process.HLTDoLocalPixelSequence = cms.Sequence()


    # Event Setup

    process.load("CalibTracker.SiPixelESProducers.siPixelGainCalibrationForHLTGPU_cfi")                 # this should be used only on GPUs, will crash otherwise
    process.load("CalibTracker.SiPixelESProducers.siPixelROCsStatusAndMappingWrapperESProducer_cfi")    # this should be used only on GPUs, will crash otherwise
    process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEFastESProducer_cfi")


    # Modules and EDAliases

    # referenced in HLTDoLocalPixelTask

    # transfer the beamspot to the gpu
    from RecoVertex.BeamSpotProducer.offlineBeamSpotToCUDA_cfi import offlineBeamSpotToCUDA as _offlineBeamSpotToCUDA
    process.hltOnlineBeamSpotToCUDA = _offlineBeamSpotToCUDA.clone(
        src = "hltOnlineBeamSpot"
    )

    # reconstruct the pixel digis and clusters on the gpu
    from RecoLocalTracker.SiPixelClusterizer.siPixelRawToClusterCUDA_cfi import siPixelRawToClusterCUDA as _siPixelRawToClusterCUDA
    process.hltSiPixelClustersCUDA = _siPixelRawToClusterCUDA.clone()
    # use the pixel channel calibrations scheme for Run 3
    run3_common.toModify(process.hltSiPixelClustersCUDA, isRun2 = False)

    # copy the pixel digis errors to the host
    from EventFilter.SiPixelRawToDigi.siPixelDigiErrorsSoAFromCUDA_cfi import siPixelDigiErrorsSoAFromCUDA as _siPixelDigiErrorsSoAFromCUDA
    process.hltSiPixelDigiErrorsSoA = _siPixelDigiErrorsSoAFromCUDA.clone(
        src = "hltSiPixelClustersCUDA"
    )

    # convert the pixel digis errors to the legacy format
    from EventFilter.SiPixelRawToDigi.siPixelDigiErrorsFromSoA_cfi import siPixelDigiErrorsFromSoA as _siPixelDigiErrorsFromSoA
    process.hltSiPixelDigiErrors = _siPixelDigiErrorsFromSoA.clone(
        digiErrorSoASrc = "hltSiPixelDigiErrorsSoA",
        UsePhase1 = True
    )

    # copy the pixel digis (except errors) and clusters to the host
    from EventFilter.SiPixelRawToDigi.siPixelDigisSoAFromCUDA_cfi import siPixelDigisSoAFromCUDA as _siPixelDigisSoAFromCUDA
    process.hltSiPixelDigisSoA = _siPixelDigisSoAFromCUDA.clone(
        src = "hltSiPixelClustersCUDA"
    )

    # convert the pixel digis (except errors) and clusters to the legacy format
    from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoA_cfi import siPixelDigisClustersFromSoA as _siPixelDigisClustersFromSoA
    process.hltSiPixelDigisClusters = _siPixelDigisClustersFromSoA.clone(
        src = "hltSiPixelDigisSoA"
    )

    # SwitchProducer wrapping the legacy pixel digis producer or an alias combining the pixel digis information converted from SoA
    process.hltSiPixelDigis = SwitchProducerCUDA(
        # legacy producer
        cpu = process.hltSiPixelDigis,
        # alias used to access products from multiple conversion modules
        cuda = cms.EDAlias(
            hltSiPixelDigisClusters = cms.VPSet(
                cms.PSet(type = cms.string("PixelDigiedmDetSetVector"))
            ),
            hltSiPixelDigiErrors = cms.VPSet(
                cms.PSet(type = cms.string("DetIdedmEDCollection")),
                cms.PSet(type = cms.string("SiPixelRawDataErroredmDetSetVector")),
                cms.PSet(type = cms.string("PixelFEDChanneledmNewDetSetVector"))
            )
        )
    )

    # SwitchProducer wrapping the legacy pixel cluster producer or an alias for the pixel clusters information converted from SoA
    process.hltSiPixelClusters = SwitchProducerCUDA(
        # legacy producer
        cpu = process.hltSiPixelClusters,
        # alias used to access products from multiple conversion modules
        cuda = cms.EDAlias(
            hltSiPixelDigisClusters = cms.VPSet(
                cms.PSet(type = cms.string("SiPixelClusteredmNewDetSetVector"))
            )
        )
    )

    # reconstruct the pixel rechits on the gpu
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDA_cfi import siPixelRecHitCUDA as _siPixelRecHitCUDA
    process.hltSiPixelRecHitsCUDA = _siPixelRecHitCUDA.clone(
        src = "hltSiPixelClustersCUDA",
        beamSpot = "hltOnlineBeamSpotToCUDA"
    )

    # SwitchProducer wrapping the legacy pixel rechit producer or the transfer of the pixel rechits to the host and the conversion from SoA
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDA_cfi import siPixelRecHitFromCUDA as _siPixelRecHitFromCUDA
    process.hltSiPixelRecHits = SwitchProducerCUDA(
        # legacy producer
        cpu = process.hltSiPixelRecHits,
        # converter to legacy format
        cuda = _siPixelRecHitFromCUDA.clone(
            pixelRecHitSrc = "hltSiPixelRecHitsCUDA",
            src = "hltSiPixelClusters"
        )
    )


    # Tasks and Sequences

    process.HLTDoLocalPixelTask = cms.Task(
          process.hltOnlineBeamSpotToCUDA,                  # transfer the beamspot to the gpu
          process.hltSiPixelClustersCUDA,                   # reconstruct the pixel digis and clusters on the gpu
          process.hltSiPixelRecHitsCUDA,                    # reconstruct the pixel rechits on the gpu
          process.hltSiPixelDigisSoA,                       # copy the pixel digis (except errors) and clusters to the host
          process.hltSiPixelDigisClusters,                  # convert the pixel digis (except errors) and clusters to the legacy format
          process.hltSiPixelDigiErrorsSoA,                  # copy the pixel digis errors to the host
          process.hltSiPixelDigiErrors,                     # convert the pixel digis errors to the legacy format
          process.hltSiPixelDigis,                          # SwitchProducer wrapping the legacy pixel digis producer or an alias combining the pixel digis information converted from SoA
          process.hltSiPixelClusters,                       # SwitchProducer wrapping the legacy pixel cluster producer or an alias for the pixel clusters information converted from SoA
          process.hltSiPixelClustersCache,                  # legacy module, used by the legacy pixel quadruplet producer
          process.hltSiPixelRecHits)                        # SwitchProducer wrapping the legacy pixel rechit producer or the transfer of the pixel rechits to the host and the conversion from SoA

    process.HLTDoLocalPixelSequence = cms.Sequence(process.HLTDoLocalPixelTask)


    # done
    return process


# customisation for running the "Patatrack" pixel track reconstruction
def customisePixelTrackReconstruction(process):

    if not 'HLTRecoPixelTracksSequence' in process.__dict__:
        return process


    # FIXME replace the Sequences with empty ones to avoid exanding them during the (re)definition of Modules and EDAliases

    process.HLTRecoPixelTracksSequence = cms.Sequence()
    process.HLTRecopixelvertexingSequence = cms.Sequence()


    # Modules and EDAliases

    # referenced in process.HLTRecoPixelTracksTask

    # cpu only: convert the pixel rechits from legacy to SoA format
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacy_cfi import siPixelRecHitSoAFromLegacy as _siPixelRecHitSoAFromLegacy
    process.hltSiPixelRecHitSoA = _siPixelRecHitSoAFromLegacy.clone(
        src = "hltSiPixelClusters",
        beamSpot = "hltOnlineBeamSpot",
        convertToLegacy = True
    )

    # build pixel ntuplets and pixel tracks in SoA format on gpu
    from RecoPixelVertexing.PixelTriplets.caHitNtupletCUDA_cfi import caHitNtupletCUDA as _caHitNtupletCUDA
    process.hltPixelTracksCUDA = _caHitNtupletCUDA.clone(
        idealConditions = False,
        pixelRecHitSrc = "hltSiPixelRecHitsCUDA",
        onGPU = True
    )
    # use quality cuts tuned for Run 2 ideal conditions for all Run 3 workflows
    run3_common.toModify(process.hltPixelTracksCUDA, idealConditions = True)

    # SwitchProducer providing the pixel tracks in SoA format on cpu
    process.hltPixelTracksSoA = SwitchProducerCUDA(
        # build pixel ntuplets and pixel tracks in SoA format on cpu
        cpu = _caHitNtupletCUDA.clone(
            idealConditions = False,
            pixelRecHitSrc = "hltSiPixelRecHitSoA",
            onGPU = False
        ),
        # transfer the pixel tracks in SoA format to the host
        cuda = cms.EDProducer("PixelTrackSoAFromCUDA",
            src = cms.InputTag("hltPixelTracksCUDA")
        )
    )
    # use quality cuts tuned for Run 2 ideal conditions for all Run 3 workflows
    run3_common.toModify(process.hltPixelTracksSoA.cpu, idealConditions = True)

    # convert the pixel tracks from SoA to legacy format
    from RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi import pixelTrackProducerFromSoA as _pixelTrackProducerFromSoA
    process.hltPixelTracks = _pixelTrackProducerFromSoA.clone(
        beamSpot = "hltOnlineBeamSpot",
        pixelRecHitLegacySrc = "hltSiPixelRecHits",
        trackSrc = "hltPixelTracksSoA"
    )


    # referenced in process.HLTRecopixelvertexingTask

    # build pixel vertices in SoA format on gpu
    from RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi import pixelVertexCUDA as _pixelVertexCUDA
    process.hltPixelVerticesCUDA = _pixelVertexCUDA.clone(
        pixelTrackSrc = "hltPixelTracksCUDA",
        onGPU = True
    )

    # build or transfer pixel vertices in SoA format on cpu
    process.hltPixelVerticesSoA = SwitchProducerCUDA(
        # build pixel vertices in SoA format on cpu
        cpu = _pixelVertexCUDA.clone(
            pixelTrackSrc = "hltPixelTracksSoA",
            onGPU = False
        ),
        # transfer the pixel vertices in SoA format to cpu
        cuda = cms.EDProducer("PixelVertexSoAFromCUDA",
            src = cms.InputTag("hltPixelVerticesCUDA")
        )
    )

    # convert the pixel vertices from SoA to legacy format
    from RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi import pixelVertexFromSoA as _pixelVertexFromSoA
    process.hltPixelVertices = _pixelVertexFromSoA.clone(
        src = "hltPixelVerticesSoA",
        TrackCollection = "hltPixelTracks",
        beamSpot = "hltOnlineBeamSpot"
    )


    # Tasks and Sequences

    process.HLTRecoPixelTracksTask = cms.Task(
          process.hltPixelTracksTrackingRegions,            # from the original sequence
          process.hltSiPixelRecHitSoA,                      # pixel rechits on cpu, converted to SoA
          process.hltPixelTracksCUDA,                       # pixel ntuplets on gpu, in SoA format
          process.hltPixelTracksSoA,                        # pixel ntuplets on cpu, in SoA format
          process.hltPixelTracks)                           # pixel tracks on cpu, in legacy format


    process.HLTRecoPixelTracksSequence = cms.Sequence(process.HLTRecoPixelTracksTask)

    process.HLTRecopixelvertexingTask = cms.Task(
          process.HLTRecoPixelTracksTask,
          process.hltPixelVerticesCUDA,                     # pixel vertices on gpu, in SoA format
          process.hltPixelVerticesSoA,                      # pixel vertices on cpu, in SoA format
          process.hltPixelVertices,                         # pixel vertices on cpu, in legacy format
          process.hltTrimmedPixelVertices)                  # from the original sequence

    process.HLTRecopixelvertexingSequence = cms.Sequence(
          process.hltPixelTracksFitter +                    # not used here, kept for compatibility with legacy sequences
          process.hltPixelTracksFilter,                     # not used here, kept for compatibility with legacy sequences
          process.HLTRecopixelvertexingTask)


    # done
    return process


# customisation for offloading the ECAL local reconstruction via CUDA if a supported gpu is present
def customiseEcalLocalReconstruction(process):

    if not 'HLTDoFullUnpackingEgammaEcalSequence' in process.__dict__:
        return process


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

    # ECAL unpacker running on gpu
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

    # SwitchProducer wrapping the legacy ECAL unpacker or the ECAL digi converter from SoA format on gpu to legacy format on cpu
    process.hltEcalDigisLegacy = process.hltEcalDigis.clone()

    process.hltEcalDigis = SwitchProducerCUDA(
        # legacy producer
        cpu = cms.EDAlias(
            hltEcalDigisLegacy = cms.VPSet(
                cms.PSet(type = cms.string("EBDigiCollection")),
                cms.PSet(type = cms.string("EEDigiCollection")),
                cms.PSet(type = cms.string("EBDetIdedmEDCollection")),
                cms.PSet(type = cms.string("EEDetIdedmEDCollection")),
                cms.PSet(type = cms.string("EBSrFlagsSorted")),
                cms.PSet(type = cms.string("EESrFlagsSorted")),
                cms.PSet(type = cms.string("EcalElectronicsIdedmEDCollection"), fromProductInstance = cms.string("EcalIntegrityBlockSizeErrors")),
                cms.PSet(type = cms.string("EcalElectronicsIdedmEDCollection"), fromProductInstance = cms.string("EcalIntegrityTTIdErrors"))
            )
        ),
        # convert ECAL digis from SoA format on gpu to legacy format on cpu
        cuda = cms.EDProducer("EcalCPUDigisProducer",
            digisInLabelEB = cms.InputTag("hltEcalDigisGPU", "ebDigis"),
            digisInLabelEE = cms.InputTag("hltEcalDigisGPU", "eeDigis"),
            digisOutLabelEB = cms.string("ebDigis"),
            digisOutLabelEE = cms.string("eeDigis"),
            produceDummyIntegrityCollections = cms.bool(True)
        )
    )

    # ECAL multifit running on gpu
    from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitProducerGPU_cfi import ecalUncalibRecHitProducerGPU as _ecalUncalibRecHitProducerGPU
    process.hltEcalUncalibRecHitGPU = _ecalUncalibRecHitProducerGPU.clone(
        digisLabelEB = ("hltEcalDigisGPU", "ebDigis"),
        digisLabelEE = ("hltEcalDigisGPU", "eeDigis"),
        shouldRunTimingComputation = False
    )

    # copy the ECAL uncalibrated rechits from gpu to cpu in SoA format
    process.hltEcalUncalibRecHitSoA = cms.EDProducer("EcalCPUUncalibRecHitProducer",
        containsTimingInformation = cms.bool(False),
        recHitsInLabelEB = cms.InputTag("hltEcalUncalibRecHitGPU", "EcalUncalibRecHitsEB"),
        recHitsInLabelEE = cms.InputTag("hltEcalUncalibRecHitGPU", "EcalUncalibRecHitsEE"),
        recHitsOutLabelEB = cms.string("EcalUncalibRecHitsEB"),
        recHitsOutLabelEE = cms.string("EcalUncalibRecHitsEE")
    )

    # SwitchProducer wrapping the legacy ECAL uncalibrated rechits producer or a converter from SoA to legacy format
    process.hltEcalUncalibRecHit = SwitchProducerCUDA(
        # legacy producer
        cpu = process.hltEcalUncalibRecHit,
        # convert the ECAL uncalibrated rechits from SoA to legacy format
        cuda = cms.EDProducer("EcalUncalibRecHitConvertGPU2CPUFormat",
            recHitsLabelGPUEB = cms.InputTag("hltEcalUncalibRecHitSoA", "EcalUncalibRecHitsEB"),
            recHitsLabelGPUEE = cms.InputTag("hltEcalUncalibRecHitSoA", "EcalUncalibRecHitsEE"),
            recHitsLabelCPUEB = cms.string("EcalUncalibRecHitsEB"),
            recHitsLabelCPUEE = cms.string("EcalUncalibRecHitsEE")
        )
    )

    # Reconstructing the ECAL calibrated rechits on gpu works, but is extremely slow.
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

    # SwitchProducer wrapping the legacy ECAL calibrated rechits producer or a converter from SoA to legacy format
    process.hltEcalRecHit = SwitchProducerCUDA(
        # legacy producer
        cpu = process.hltEcalRecHit,
        # convert the ECAL calibrated rechits from SoA to legacy format
        cuda = cms.EDProducer("EcalRecHitConvertGPU2CPUFormat",
            recHitsLabelGPUEB = cms.InputTag("hltEcalRecHitSoA", "EcalRecHitsEB"),
            recHitsLabelGPUEE = cms.InputTag("hltEcalRecHitSoA", "EcalRecHitsEE"),
            recHitsLabelCPUEB = cms.string("EcalRecHitsEB"),
            recHitsLabelCPUEE = cms.string("EcalRecHitsEE"),
        )
    """

    
    # SwitchProducer wrapping the legacy ECAL rechits producer
    # the gpu unpacker does not produce the TPs used for the recovery, so the SwitchProducer alias does not provide them:
    #   - the cpu uncalibrated rechit producer may mark them for recovery, read the TPs explicitly from the legacy unpacker
    #   - the gpu uncalibrated rechit producer does not flag them for recovery, so the TPs are not necessary
    process.hltEcalRecHit = SwitchProducerCUDA(
        cpu = process.hltEcalRecHit.clone(
            triggerPrimitiveDigiCollection = cms.InputTag('hltEcalDigisLegacy', 'EcalTriggerPrimitives')
        ),
        cuda = process.hltEcalRecHit.clone(
            triggerPrimitiveDigiCollection = cms.InputTag('unused')
        )
    )

    # Tasks and Sequences

    process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask = cms.Task(
        process.hltEcalDigisGPU,                            # unpack ECAL digis on gpu
        process.hltEcalDigisLegacy,                         # legacy producer, referenced in the SwitchProducer
        process.hltEcalDigis,                               # SwitchProducer
        process.hltEcalUncalibRecHitGPU,                    # run ECAL local reconstruction and multifit on gpu
        process.hltEcalUncalibRecHitSoA,                    # needed by hltEcalPhiSymFilter - copy to host
        process.hltEcalUncalibRecHit,                       # needed by hltEcalPhiSymFilter - convert to legacy format
      # process.hltEcalRecHitGPU,                           # make ECAL calibrated rechits on gpu
      # process.hltEcalRecHitSoA,                           # copy to host
        process.hltEcalDetIdToBeRecovered,                  # legacy producer
        process.hltEcalRecHit)                              # legacy producer

    process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence = cms.Sequence(
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask)

    process.HLTPreshowerTask = cms.Task(
        process.hltEcalPreshowerDigis,                      # unpack ECAL preshower digis on the host
        process.hltEcalPreshowerRecHit)                     # build ECAL preshower rechits on the host

    process.HLTPreshowerSequence = cms.Sequence(process.HLTPreshowerTask)

    process.HLTDoFullUnpackingEgammaEcalTask = cms.Task(
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask,
        process.HLTPreshowerTask)

    process.HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence(
        process.HLTDoFullUnpackingEgammaEcalTask)

    process.HLTDoFullUnpackingEgammaEcalMFSequence = cms.Sequence(
        process.HLTDoFullUnpackingEgammaEcalTask)


    # done
    return process

# customisation for offloading the HCAL local reconstruction via CUDA if a supported gpu is present
def customiseHcalLocalReconstruction(process):

    if not 'HLTDoLocalHcalSequence' in process.__dict__:
        return process


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
        maxChannelsF3HB = cms.uint32(10000)
    )

    # run the HCAL local reconstruction (including Method 0 and MAHI) on gpu
    from RecoLocalCalo.HcalRecProducers.hbheRecHitProducerGPU_cfi import hbheRecHitProducerGPU as _hbheRecHitProducerGPU
    process.hltHbherecoGPU = _hbheRecHitProducerGPU.clone(
        digisLabelF01HE = "hltHcalDigisGPU",
        digisLabelF5HB = "hltHcalDigisGPU",
        digisLabelF3HB = "hltHcalDigisGPU",
        recHitsLabelM0HBHE = ""
    )

    # transfer the HCAL rechits to the cpu, and convert them to the legacy format
    from RecoLocalCalo.HcalRecProducers.hcalCPURecHitsProducer_cfi import hcalCPURecHitsProducer as _hcalCPURecHitsProducer
    process.hltHbherecoFromGPU = _hcalCPURecHitsProducer.clone(
        recHitsM0LabelIn = "hltHbherecoGPU",
        recHitsM0LabelOut = "",
        recHitsLegacyLabelOut = ""
    )

    # SwitchProducer between the legacy producer and the copy from gpu with conversion
    process.hltHbhereco = SwitchProducerCUDA(
        # legacy producer
        cpu = process.hltHbhereco.clone(),
        # alias to the rechits converted to legacy format
        cuda = cms.EDAlias(
            hltHbherecoFromGPU = cms.VPSet(
                cms.PSet(type = cms.string("HBHERecHitsSorted"))
            )
        )
    )


    # Tasks and Sequences

    process.HLTDoLocalHcalTask = cms.Task(
        process.hltHcalDigis,                               # legacy producer, unpack HCAL digis on cpu
        process.hltHcalDigisGPU,                            # copy to gpu and convert to SoA format
        process.hltHbherecoGPU,                             # run the HCAL local reconstruction (including Method 0 and MAHI) on gpu
        process.hltHbherecoFromGPU,                         # transfer the HCAL rechits to the cpu, and convert them to the legacy format
        process.hltHbhereco,                                # SwitchProducer between the legacy producer and the copy from gpu with conversion
        process.hltHfprereco,                               # legacy producer
        process.hltHfreco,                                  # legacy producer
        process.hltHoreco)                                  # legacy producer

    process.HLTDoLocalHcalSequence = cms.Sequence(
        process.HLTDoLocalHcalTask)

    process.HLTStoppedHSCPLocalHcalRecoTask = cms.Task(
        process.hltHcalDigis,                               # legacy producer, unpack HCAL digis on cpu
        process.hltHcalDigisGPU,                            # copy to gpu and convert to SoA format
        process.hltHbherecoGPU,                             # run the HCAL local reconstruction (including Method 0 and MAHI) on gpu
        process.hltHbherecoFromGPU,                         # transfer the HCAL rechits to the cpu, and convert them to the legacy format
        process.hltHbhereco)                                # SwitchProducer between the legacy producer and the copy from gpu with conversion

    process.HLTStoppedHSCPLocalHcalReco = cms.Sequence(
        process.HLTStoppedHSCPLocalHcalRecoTask)


    # done
    return process


# customisation for running the Patatrack reconstruction, with automatic offload via CUDA when a supported gpu is available
def customizeHLTforPatatrack(process):
    process = customiseCommon(process)
    process = customisePixelLocalReconstruction(process)
    process = customisePixelTrackReconstruction(process)
    process = customiseEcalLocalReconstruction(process)
    process = customiseHcalLocalReconstruction(process)
    return process


def _addConsumerPath(process):
    # add to a path all consumers and the tasks that define the producers
    process.Consumer = cms.Path(
        process.HLTBeginSequence +
        process.hltPixelConsumer +
        process.hltEcalConsumer +
        process.hltHbheConsumer,
        process.HLTDoLocalPixelTask,
        process.HLTRecoPixelTracksTask,
        process.HLTRecopixelvertexingTask,
        process.HLTDoFullUnpackingEgammaEcalTask,
        process.HLTDoLocalHcalTask,
    )

    if 'HLTSchedule' in process.__dict__:
        process.HLTSchedule.append(process.Consumer)
    if process.schedule is not None:
        process.schedule.append(process.Consumer)

    # done
    return process


def consumeGPUSoAProducts(process):
    # consume the Pixel tracks and vertices on the GPU in SoA format
    process.hltPixelConsumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring( 'hltPixelTracksCUDA', 'hltPixelVerticesCUDA' )
    )

    # consume the ECAL uncalibrated rechits on the GPU in SoA format
    process.hltEcalConsumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring( 'hltEcalUncalibRecHitGPU' )
    )

    # consume the HCAL rechits on the GPU in SoA format
    process.hltHbheConsumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring( 'hltHbherecoGPU' )
    )

    # add to a path all consumers and the tasks that define the producers
    process = _addConsumerPath(process)

    # done
    return process


def consumeCPUSoAProducts(process):
    # consume the Pixel tracks and vertices on the CPU in SoA format
    process.hltPixelConsumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring( 'hltPixelTracksSoA', 'hltPixelVerticesSoA' )
    )

    # consume the ECAL uncalibrated rechits on the CPU in SoA format
    process.hltEcalConsumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring( 'hltEcalUncalibRecHitSoA' )
    )

    # consume the HCAL rechits on the CPU in legacy format
    process.hltHbheConsumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring( 'hltHbhereco' )
    )

    # add to a path all consumers and the tasks that define the producers
    process = _addConsumerPath(process)

    # done
    return process

def consumeCPULegacyProducts(process):
    # consume the Pixel tracks and vertices on the CPU in legacy format
    process.hltPixelConsumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring( 'hltPixelTracks', 'hltPixelVertices' )
    )

    # consume the ECAL runcalibrated echits on the CPU in legacy format
    process.hltEcalConsumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring( 'hltEcalUncalibRecHit' )
    )

    # consume the HCAL rechits on the CPU in legacy format
    process.hltHbheConsumer = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring( 'hltHbhereco' )
    )

    # add to a path all consumers and the tasks that define the producers
    process = _addConsumerPath(process)

    # done
    return process
