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


# customisation for running the Patatrack reconstruction, common parts
def customiseCommon(process):

    # Services

    process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
    if 'MessageLogger' in process.__dict__:
        process.MessageLogger.CUDAService = cms.untracked.PSet()

    # NVProfilerService is broken in CMSSW 12.0.x and later
    #process.load("HeterogeneousCore.CUDAServices.NVProfilerService_cfi")


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

    if 'Status_OnCPU' in process.__dict__:
        replace_with(process.Status_OnCPU, cms.Path(process.statusOnGPU + ~process.statusOnGPUFilter))
    else:
        process.Status_OnCPU = cms.Path(process.statusOnGPU + ~process.statusOnGPUFilter)
        if process.schedule is not None:
            process.schedule.append(process.Status_OnCPU)

    if 'Status_OnGPU' in process.__dict__:
        replace_with(process.Status_OnGPU, cms.Path(process.statusOnGPU + process.statusOnGPUFilter))
    else:
        process.Status_OnGPU = cms.Path(process.statusOnGPU + process.statusOnGPUFilter)
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

    # FIXME replace the Sequences with empty ones to avoid expanding them during the (re)definition of Modules and EDAliases

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
    process.hltSiPixelClustersCUDA = _siPixelRawToClusterCUDA.clone(
        # use the same thresholds as the legacy module
        clusterThreshold_layer1 = process.hltSiPixelClusters.ClusterThreshold_L1,
        clusterThreshold_otherLayers = process.hltSiPixelClusters.ClusterThreshold
    )
    # use the pixel channel calibrations scheme for Run 3
    run3_common.toModify(process.hltSiPixelClustersCUDA, isRun2 = False)

    # copy the pixel digis errors to the host
    from EventFilter.SiPixelRawToDigi.siPixelDigiErrorsSoAFromCUDA_cfi import siPixelDigiErrorsSoAFromCUDA as _siPixelDigiErrorsSoAFromCUDA
    process.hltSiPixelDigiErrorsSoA = _siPixelDigiErrorsSoAFromCUDA.clone(
        src = "hltSiPixelClustersCUDA"
    )

    # copy the pixel digis (except errors) and clusters to the host
    from EventFilter.SiPixelRawToDigi.siPixelDigisSoAFromCUDA_cfi import siPixelDigisSoAFromCUDA as _siPixelDigisSoAFromCUDA
    process.hltSiPixelDigisSoA = _siPixelDigisSoAFromCUDA.clone(
        src = "hltSiPixelClustersCUDA"
    )

    # reconstruct the pixel digis on the cpu
    process.hltSiPixelDigisLegacy = process.hltSiPixelDigis.clone()

    # SwitchProducer wrapping a subset of the legacy pixel digis producer, or the conversion of the pixel digis errors to the legacy format
    from EventFilter.SiPixelRawToDigi.siPixelDigiErrorsFromSoA_cfi import siPixelDigiErrorsFromSoA as _siPixelDigiErrorsFromSoA
    process.hltSiPixelDigis = SwitchProducerCUDA(
        # legacy producer
        cpu = cms.EDAlias(
            hltSiPixelDigisLegacy = cms.VPSet(
                cms.PSet(type = cms.string("DetIdedmEDCollection")),
                cms.PSet(type = cms.string("SiPixelRawDataErroredmDetSetVector")),
                cms.PSet(type = cms.string("PixelFEDChanneledmNewDetSetVector"))
            )
        ),
        # conversion from SoA to legacy format
        cuda = _siPixelDigiErrorsFromSoA.clone(
            digiErrorSoASrc = "hltSiPixelDigiErrorsSoA",
            UsePhase1 = True
        )
    )

    # reconstruct the pixel clusters on the cpu
    process.hltSiPixelClustersLegacy = process.hltSiPixelClusters.clone(
        src = "hltSiPixelDigisLegacy"
    )

    # SwitchProducer wrapping a subset of the legacy pixel cluster producer, or the conversion of the pixel digis (except errors) and clusters to the legacy format
    from RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoA_cfi import siPixelDigisClustersFromSoA as _siPixelDigisClustersFromSoA
    process.hltSiPixelClusters = SwitchProducerCUDA(
        # legacy producer
        cpu = cms.EDAlias(
            hltSiPixelClustersLegacy = cms.VPSet(
                cms.PSet(type = cms.string("SiPixelClusteredmNewDetSetVector"))
            )
        ),
        # conversion from SoA to legacy format
        cuda = _siPixelDigisClustersFromSoA.clone(
            src = "hltSiPixelDigisSoA",
            produceDigis = False,
            storeDigis = False,
            # use the same thresholds as the legacy module
            clusterThreshold_layer1 = process.hltSiPixelClusters.ClusterThreshold_L1,
            clusterThreshold_otherLayers = process.hltSiPixelClusters.ClusterThreshold
        )
    )

    # reconstruct the pixel rechits on the gpu
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDA_cfi import siPixelRecHitCUDA as _siPixelRecHitCUDA
    process.hltSiPixelRecHitsCUDA = _siPixelRecHitCUDA.clone(
        src = "hltSiPixelClustersCUDA",
        beamSpot = "hltOnlineBeamSpotToCUDA"
    )

    # cpu only: produce the pixel rechits in SoA and legacy format, from the legacy clusters
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacy_cfi import siPixelRecHitSoAFromLegacy as _siPixelRecHitSoAFromLegacy
    process.hltSiPixelRecHitSoA = _siPixelRecHitSoAFromLegacy.clone(
        src = "hltSiPixelClusters",
        beamSpot = "hltOnlineBeamSpot",
        convertToLegacy = True
    )

    # SwitchProducer wrapping the legacy pixel rechit producer or the transfer of the pixel rechits to the host and the conversion from SoA
    from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDA_cfi import siPixelRecHitFromCUDA as _siPixelRecHitFromCUDA
    process.hltSiPixelRecHits = SwitchProducerCUDA(
        # legacy producer
        cpu = cms.EDAlias(
           hltSiPixelRecHitSoA = cms.VPSet(
                cms.PSet(type = cms.string("SiPixelRecHitedmNewDetSetVector")),
                cms.PSet(type = cms.string("uintAsHostProduct"))
            )
        ),
        # conversion from SoA to legacy format
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
          process.hltSiPixelDigiErrorsSoA,                  # copy the pixel digis errors to the host
          process.hltSiPixelDigisLegacy,                    # legacy pixel digis producer
          process.hltSiPixelDigis,                          # SwitchProducer wrapping a subset of the legacy pixel digis producer, or the conversion of the pixel digis errors from SoA
          process.hltSiPixelClustersLegacy,                 # legacy pixel cluster producer
          process.hltSiPixelClusters,                       # SwitchProducer wrapping a subset of the legacy pixel cluster producer, or the conversion of the pixel digis (except errors) and clusters from SoA
          process.hltSiPixelClustersCache,                  # legacy module, used by the legacy pixel quadruplet producer
          process.hltSiPixelRecHitSoA,                      # pixel rechits on cpu, in SoA & legacy format
          process.hltSiPixelRecHits)                        # SwitchProducer wrapping the legacy pixel rechit producer or the transfer of the pixel rechits to the host and the conversion from SoA

    process.HLTDoLocalPixelSequence = cms.Sequence(process.HLTDoLocalPixelTask)


    # workaround for AlCa paths
    for AlCaPathName in ['AlCa_LumiPixelsCounts_Random_v1', 'AlCa_LumiPixelsCounts_ZeroBias_v1']:
        if AlCaPathName in process.__dict__:
            AlCaPath = getattr(process, AlCaPathName)
            # replace hltSiPixelDigis+hltSiPixelClusters with HLTDoLocalPixelSequence
            hasSiPixelDigis, hasSiPixelClusters = False, False
            for (itemLabel, itemName) in AlCaPath.directDependencies():
                if itemLabel != 'modules': continue
                if itemName == 'hltSiPixelDigis': hasSiPixelDigis = True
                elif itemName == 'hltSiPixelClusters': hasSiPixelClusters = True
            if hasSiPixelDigis and hasSiPixelClusters:
                AlCaPath.remove(process.hltSiPixelClusters)
                AlCaPath.replace(process.hltSiPixelDigis, process.HLTDoLocalPixelSequence)


    # done
    return process


# customisation for running the "Patatrack" pixel track reconstruction
def customisePixelTrackReconstruction(process):

    if not 'HLTRecoPixelTracksSequence' in process.__dict__:
        return process

    hasHLTPixelVertexReco = 'HLTRecopixelvertexingSequence' in process.__dict__

    # FIXME replace the Sequences with empty ones to avoid expanding them during the (re)definition of Modules and EDAliases

    process.HLTRecoPixelTracksSequence = cms.Sequence()
    if hasHLTPixelVertexReco:
        process.HLTRecopixelvertexingSequence = cms.Sequence()


    # Modules and EDAliases

    # referenced in process.HLTRecoPixelTracksTask

    # build pixel ntuplets and pixel tracks in SoA format on gpu
    from RecoPixelVertexing.PixelTriplets.pixelTracksCUDA_cfi import pixelTracksCUDA as _pixelTracksCUDA
    process.hltPixelTracksCUDA = _pixelTracksCUDA.clone(
        idealConditions = False,
        pixelRecHitSrc = "hltSiPixelRecHitsCUDA",
        onGPU = True
    )
    # use quality cuts tuned for Run 2 ideal conditions for all Run 3 workflows
    run3_common.toModify(process.hltPixelTracksCUDA, idealConditions = True)

    # SwitchProducer providing the pixel tracks in SoA format on cpu
    from RecoPixelVertexing.PixelTrackFitting.pixelTracksSoA_cfi import pixelTracksSoA as _pixelTracksSoA
    process.hltPixelTracksSoA = SwitchProducerCUDA(
        # build pixel ntuplets and pixel tracks in SoA format on cpu
        cpu = _pixelTracksCUDA.clone(
            idealConditions = False,
            pixelRecHitSrc = "hltSiPixelRecHitSoA",
            onGPU = False
        ),
        # transfer the pixel tracks in SoA format to the host
        cuda = _pixelTracksSoA.clone(
            src = "hltPixelTracksCUDA"
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
    if hasHLTPixelVertexReco:

        # build pixel vertices in SoA format on gpu
        from RecoPixelVertexing.PixelVertexFinding.pixelVerticesCUDA_cfi import pixelVerticesCUDA as _pixelVerticesCUDA
        process.hltPixelVerticesCUDA = _pixelVerticesCUDA.clone(
            pixelTrackSrc = "hltPixelTracksCUDA",
            onGPU = True
        )

        # build or transfer pixel vertices in SoA format on cpu
        from RecoPixelVertexing.PixelVertexFinding.pixelVerticesSoA_cfi import pixelVerticesSoA as _pixelVerticesSoA
        process.hltPixelVerticesSoA = SwitchProducerCUDA(
            # build pixel vertices in SoA format on cpu
            cpu = _pixelVerticesCUDA.clone(
                pixelTrackSrc = "hltPixelTracksSoA",
                onGPU = False
            ),
            # transfer the pixel vertices in SoA format to cpu
            cuda = _pixelVerticesSoA.clone(
                src = "hltPixelVerticesCUDA"
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
          process.hltPixelTracksCUDA,                       # pixel ntuplets on gpu, in SoA format
          process.hltPixelTracksSoA,                        # pixel ntuplets on cpu, in SoA format
          process.hltPixelTracks)                           # pixel tracks on cpu, in legacy format


    process.HLTRecoPixelTracksSequence = cms.Sequence(process.HLTRecoPixelTracksTask)

    if hasHLTPixelVertexReco:
        process.HLTRecopixelvertexingTask = cms.Task(
              process.HLTRecoPixelTracksTask,
              process.hltPixelVerticesCUDA,                 # pixel vertices on gpu, in SoA format
              process.hltPixelVerticesSoA,                  # pixel vertices on cpu, in SoA format
              process.hltPixelVertices,                     # pixel vertices on cpu, in legacy format
              process.hltTrimmedPixelVertices)              # from the original sequence

        process.HLTRecopixelvertexingSequence = cms.Sequence(
              process.hltPixelTracksFitter +                # not used here, kept for compatibility with legacy sequences
              process.hltPixelTracksFilter,                 # not used here, kept for compatibility with legacy sequences
              process.HLTRecopixelvertexingTask)


    # done
    return process


# customisation for offloading the ECAL local reconstruction via CUDA if a supported gpu is present
def customiseEcalLocalReconstruction(process):

    hasHLTEcalPreshowerSeq = any(seq in process.__dict__ for seq in ['HLTDoFullUnpackingEgammaEcalMFSequence', 'HLTDoFullUnpackingEgammaEcalSequence'])
    if not (hasHLTEcalPreshowerSeq or 'HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence' in process.__dict__):
        return process

    # FIXME replace the Sequences with empty ones to avoid expanding them during the (re)definition of Modules and EDAliases

    process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence = cms.Sequence()
    if hasHLTEcalPreshowerSeq:
        process.HLTDoFullUnpackingEgammaEcalMFSequence = cms.Sequence()
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
    from EventFilter.EcalRawToDigi.ecalRawToDigiGPU_cfi import ecalRawToDigiGPU as _ecalRawToDigiGPU
    process.hltEcalDigisGPU = _ecalRawToDigiGPU.clone()

    # SwitchProducer wrapping the legacy ECAL unpacker or the ECAL digi converter from SoA format on gpu to legacy format on cpu
    process.hltEcalDigisLegacy = process.hltEcalDigis.clone()
    from EventFilter.EcalRawToDigi.ecalCPUDigisProducer_cfi import ecalCPUDigisProducer as _ecalCPUDigisProducer

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
                cms.PSet(type = cms.string("EcalElectronicsIdedmEDCollection"), fromProductInstance = cms.string("EcalIntegrityTTIdErrors")),
                cms.PSet(type = cms.string("EcalElectronicsIdedmEDCollection"), fromProductInstance = cms.string("EcalIntegrityZSXtalIdErrors")),
                cms.PSet(type = cms.string("EcalPnDiodeDigisSorted")),
                cms.PSet(type = cms.string("EcalPseudoStripInputDigisSorted"), fromProductInstance = cms.string("EcalPseudoStripInputs")),
                cms.PSet(type = cms.string("EcalTriggerPrimitiveDigisSorted"), fromProductInstance = cms.string("EcalTriggerPrimitives")),
            )
        ),
        # convert ECAL digis from SoA format on gpu to legacy format on cpu
        cuda = _ecalCPUDigisProducer.clone(
            digisInLabelEB = ("hltEcalDigisGPU", "ebDigis"),
            digisInLabelEE = ("hltEcalDigisGPU", "eeDigis"),
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
    from RecoLocalCalo.EcalRecProducers.ecalCPUUncalibRecHitProducer_cfi import ecalCPUUncalibRecHitProducer as _ecalCPUUncalibRecHitProducer
    process.hltEcalUncalibRecHitSoA = _ecalCPUUncalibRecHitProducer.clone(
        recHitsInLabelEB = ("hltEcalUncalibRecHitGPU", "EcalUncalibRecHitsEB"),
        recHitsInLabelEE = ("hltEcalUncalibRecHitGPU", "EcalUncalibRecHitsEE"),
    )

    # SwitchProducer wrapping the legacy ECAL uncalibrated rechits producer or a converter from SoA to legacy format
    from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitConvertGPU2CPUFormat_cfi import ecalUncalibRecHitConvertGPU2CPUFormat as _ecalUncalibRecHitConvertGPU2CPUFormat
    process.hltEcalUncalibRecHit = SwitchProducerCUDA(
        # legacy producer
        cpu = process.hltEcalUncalibRecHit,
        # convert the ECAL uncalibrated rechits from SoA to legacy format
        cuda = _ecalUncalibRecHitConvertGPU2CPUFormat.clone(
            recHitsLabelGPUEB = ("hltEcalUncalibRecHitSoA", "EcalUncalibRecHitsEB"),
            recHitsLabelGPUEE = ("hltEcalUncalibRecHitSoA", "EcalUncalibRecHitsEE"),
        )
    )

    # Reconstructing the ECAL calibrated rechits on gpu works, but is extremely slow.
    # Disable it for the time being, until the performance has been addressed.
    """
    from RecoLocalCalo.EcalRecProducers.ecalRecHitGPU_cfi import ecalRecHitGPU as _ecalRecHitGPU
    process.hltEcalRecHitGPU = _ecalRecHitGPU.clone(
        uncalibrecHitsInLabelEB = ("hltEcalUncalibRecHitGPU","EcalUncalibRecHitsEB"),
        uncalibrecHitsInLabelEE = ("hltEcalUncalibRecHitGPU","EcalUncalibRecHitsEE"),
    )

    from RecoLocalCalo.EcalRecProducers.ecalCPURecHitProducer_cfi import ecalCPURecHitProducer as _ecalCPURecHitProducer
    process.hltEcalRecHitSoA = _ecalCPURecHitProducer.clone(
        recHitsInLabelEB = ("hltEcalRecHitGPU", "EcalRecHitsEB"),
        recHitsInLabelEE = ("hltEcalRecHitGPU", "EcalRecHitsEE"),
    )

    # SwitchProducer wrapping the legacy ECAL calibrated rechits producer or a converter from SoA to legacy format
    from RecoLocalCalo.EcalRecProducers.ecalRecHitConvertGPU2CPUFormat_cfi import ecalRecHitConvertGPU2CPUFormat as _ecalRecHitConvertGPU2CPUFormat
    process.hltEcalRecHit = SwitchProducerCUDA(
        # legacy producer
        cpu = process.hltEcalRecHit,
        # convert the ECAL calibrated rechits from SoA to legacy format
        cuda = _ecalRecHitConvertGPU2CPUFormat.clone(
            recHitsLabelGPUEB = ("hltEcalRecHitSoA", "EcalRecHitsEB"),
            recHitsLabelGPUEE = ("hltEcalRecHitSoA", "EcalRecHitsEE"),
        )
    )
    """

    # SwitchProducer wrapping the legacy ECAL rechits producer
    # the gpu unpacker does not produce the TPs used for the recovery, so the SwitchProducer alias does not provide them:
    #   - the cpu uncalibrated rechit producer may mark them for recovery, read the TPs explicitly from the legacy unpacker
    #   - the gpu uncalibrated rechit producer does not flag them for recovery, so the TPs are not necessary
    process.hltEcalRecHit = SwitchProducerCUDA(
        cpu = process.hltEcalRecHit.clone(
            triggerPrimitiveDigiCollection = ('hltEcalDigisLegacy', 'EcalTriggerPrimitives')
        ),
        cuda = process.hltEcalRecHit.clone(
            triggerPrimitiveDigiCollection = 'unused'
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

    if hasHLTEcalPreshowerSeq:
        process.HLTPreshowerTask = cms.Task(
            process.hltEcalPreshowerDigis,                  # unpack ECAL preshower digis on the host
            process.hltEcalPreshowerRecHit)                 # build ECAL preshower rechits on the host

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

    hasHLTDoLocalHcalSeq = 'HLTDoLocalHcalSequence' in process.__dict__
    if not (hasHLTDoLocalHcalSeq or 'HLTStoppedHSCPLocalHcalReco' in process.__dict__):
        return process

    # FIXME replace the Sequences with empty ones to avoid expanding them during the (re)definition of Modules and EDAliases

    if hasHLTDoLocalHcalSeq:
        process.HLTDoLocalHcalSequence = cms.Sequence()
    process.HLTStoppedHSCPLocalHcalReco = cms.Sequence()


    # Event Setup

    process.load("EventFilter.HcalRawToDigi.hcalElectronicsMappingGPUESProducer_cfi")

    process.load("RecoLocalCalo.HcalRecProducers.hcalChannelQualityGPUESProducer_cfi")
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
    from EventFilter.HcalRawToDigi.hcalDigisProducerGPU_cfi import hcalDigisProducerGPU as _hcalDigisProducerGPU
    process.hltHcalDigisGPU = _hcalDigisProducerGPU.clone(
        hbheDigisLabel = "hltHcalDigis",
        qie11DigiLabel = "hltHcalDigis",
        digisLabelF01HE = "",
        digisLabelF5HB = "",
        digisLabelF3HB = ""
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
    if hasHLTDoLocalHcalSeq:
        process.HLTDoLocalHcalTask = cms.Task(
            process.hltHcalDigis,                           # legacy producer, unpack HCAL digis on cpu
            process.hltHcalDigisGPU,                        # copy to gpu and convert to SoA format
            process.hltHbherecoGPU,                         # run the HCAL local reconstruction (including Method 0 and MAHI) on gpu
            process.hltHbherecoFromGPU,                     # transfer the HCAL rechits to the cpu, and convert them to the legacy format
            process.hltHbhereco,                            # SwitchProducer between the legacy producer and the copy from gpu with conversion
            process.hltHfprereco,                           # legacy producer
            process.hltHfreco,                              # legacy producer
            process.hltHoreco)                              # legacy producer

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


# customisation to enable pixel triplets instead of quadruplets
def enablePatatrackPixelTriplets(process):

  if 'hltPixelTracksCUDA' in process.__dict__:
      # configure GPU pixel tracks for triplets
      process.hltPixelTracksCUDA.minHitsPerNtuplet = 3
      process.hltPixelTracksCUDA.includeJumpingForwardDoublets = True

  if 'hltPixelTracksSoA' in process.__dict__:
      # configure CPU pixel tracks for triplets
      process.hltPixelTracksSoA.cpu.minHitsPerNtuplet = 3
      process.hltPixelTracksSoA.cpu.includeJumpingForwardDoublets = True

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


# customisation for running the Patatrack triplets reconstruction, with automatic offload via CUDA when a supported gpu is available
def customizeHLTforPatatrackTriplets(process):
    process = customiseCommon(process)
    process = customisePixelLocalReconstruction(process)
    process = customisePixelTrackReconstruction(process)
    process = customiseEcalLocalReconstruction(process)
    process = customiseHcalLocalReconstruction(process)
    process = enablePatatrackPixelTriplets(process)
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
