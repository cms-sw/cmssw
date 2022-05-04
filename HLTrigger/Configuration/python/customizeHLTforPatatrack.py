import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
from HLTrigger.Configuration.common import *
from Configuration.Eras.Modifier_run3_common_cff import run3_common

def _load_if_missing(process, label, config, check = True):
    '''
    Utility to load file "config" in the process, if the process does not already hold a module named "label"
     - if "check" is true and "process.label" does not exist after loading "config", an exception is thrown
    Example: _load_if_missing(process, 'SomeESProducer', 'SubSystem.Package.SomeESProducer_cfi')
    '''
    if label not in process.__dict__:
        process.load(config)
    if check and label not in process.__dict__:
        raise Exception('process does not have a module labelled "'+label+'" after loading "'+config+'"')

def _clone(_obj, _label, _config, _original_label = None, _only_if_missing = False, **kwargs):
    '''
    Utility to add "_label" to "_obj" using a clone of module "_original_label" from file "_config"
     - if "_original_label" is not specified, it is set equal to "_label"
     - if "_only_if_missing" is true and "_obj._label" exists, nothing is done
    Example: _clone(process, 'hltMyFilter', 'SubSystem.Package.hltSomeFilter_cfi', 'hltSomeFilter',
               _only_if_missing = False, modArg1 = val1, modArg2 = val2)
    '''
    if not _only_if_missing or not hasattr(_obj, _label):
        if _original_label is None:
            _original_label = _label
        _module = __import__(_config, globals(), locals(), [ _original_label ], 0)
        setattr(_obj, _label, getattr(_module, _original_label).clone(**kwargs))

def _clone_if_missing(_obj, _label, _config, _original_label = None, **kwargs):
    '''
    Wrapper of _clone with _only_if_missing=True (for examples, see _clone)
    '''
    _clone(_obj, _label, _config, _original_label, _only_if_missing = True, **kwargs)


# customisation for running the Patatrack reconstruction, common parts
def customiseCommon(process):

    # Services

    process.load('Configuration.StandardSequences.Accelerators_cff')

#    # NVProfilerService is broken in CMSSW 12.0.x and later
#    _load_if_missing(process, 'NVProfilerService', 'HeterogeneousCore.CUDAServices.NVProfilerService_cfi')


    # Paths and EndPaths

    # the hltGetConditions module would force gpu-specific ESProducers to run even if no supported gpu is present
    if 'hltGetConditions' in process.__dict__:
        del process.hltGetConditions

    # produce a boolean to track if the events ar being processed on gpu (true) or cpu (false)
    if 'statusOnGPU' not in process.__dict__:
        process.statusOnGPU = SwitchProducerCUDA(
            cpu  = cms.EDProducer("BooleanProducer", value = cms.bool(False))
        )

    if not hasattr(process.statusOnGPU, 'cuda'):
        process.statusOnGPU.cuda = cms.EDProducer("BooleanProducer", value = cms.bool(True))

    if 'statusOnGPUFilter' not in process.__dict__:
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
    if 'hltOutputScoutingCaloMuon' in process.__dict__ and 'hltPreScoutingCaloMuonOutput' in process.__dict__ and not 'hltPreScoutingCaloMuonOutputSmart' in process.__dict__:
        process.hltPreScoutingCaloMuonOutputSmart = cms.EDFilter( "TriggerResultsFilter",
            l1tIgnoreMaskAndPrescale = cms.bool( False ),
            l1tResults = cms.InputTag( "" ),
            hltResults = cms.InputTag( 'TriggerResults','','@currentProcess' ),
            triggerConditions = process.hltOutputScoutingCaloMuon.SelectEvents.SelectEvents,
            throw = cms.bool( True )
        )
        insert_modules_after(process, process.hltPreScoutingCaloMuonOutput, process.hltPreScoutingCaloMuonOutputSmart)

    # make the ScoutingPFOutput endpath compatible with using Tasks in the Scouting paths
    if 'hltOutputScoutingPF' in process.__dict__ and 'hltPreScoutingPFOutput' in process.__dict__ and not 'hltPreScoutingPFOutputSmart' in process.__dict__:
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

    _load_if_missing(process, 'PixelCPEFastESProducer', 'RecoLocalTracker.SiPixelRecHits.PixelCPEFastESProducer_cfi')
    # these 2 modules should be used only on GPUs, will crash otherwise
    _load_if_missing(process, 'siPixelGainCalibrationForHLTGPU', 'CalibTracker.SiPixelESProducers.siPixelGainCalibrationForHLTGPU_cfi')
    _load_if_missing(process, 'siPixelROCsStatusAndMappingWrapperESProducer', 'CalibTracker.SiPixelESProducers.siPixelROCsStatusAndMappingWrapperESProducer_cfi')


    # Modules and EDAliases
    # referenced in HLTDoLocalPixelTask

    # transfer the beamspot to the gpu
    _clone_if_missing(process, 'hltOnlineBeamSpotToCUDA', 'RecoVertex.BeamSpotProducer.offlineBeamSpotToCUDA_cfi', 'offlineBeamSpotToCUDA',
        src = 'hltOnlineBeamSpot'
    )

    if 'hltSiPixelClustersCUDA' not in process.__dict__:
        # reconstruct the pixel digis and clusters on the gpu
        _hltSiPixelClustersLegacyLabel = 'hltSiPixelClustersLegacy' if 'hltSiPixelClustersLegacy' in process.__dict__ else 'hltSiPixelClusters'

        _clone(process, 'hltSiPixelClustersCUDA', 'RecoLocalTracker.SiPixelClusterizer.siPixelRawToClusterCUDA_cfi', 'siPixelRawToClusterCUDA',
            # use the same thresholds as the legacy module
            clusterThreshold_layer1 = getattr(process, _hltSiPixelClustersLegacyLabel).ClusterThreshold_L1,
            clusterThreshold_otherLayers = getattr(process, _hltSiPixelClustersLegacyLabel).ClusterThreshold,
        )
        # use the pixel channel calibrations scheme for Run 3
        run3_common.toModify(process.hltSiPixelClustersCUDA, isRun2 = False)

    # copy the pixel digis errors to the host
    _clone_if_missing(process, 'hltSiPixelDigiErrorsSoA', 'EventFilter.SiPixelRawToDigi.siPixelDigiErrorsSoAFromCUDA_cfi', 'siPixelDigiErrorsSoAFromCUDA',
        src = 'hltSiPixelClustersCUDA'
    )

    # copy the pixel digis (except errors) and clusters to the host
    _clone_if_missing(process, 'hltSiPixelDigisSoA', 'EventFilter.SiPixelRawToDigi.siPixelDigisSoAFromCUDA_cfi', 'siPixelDigisSoAFromCUDA',
        src = 'hltSiPixelClustersCUDA'
    )

    # SwitchProducer: hltSiPixelDigis
    if not isinstance(process.hltSiPixelDigis, SwitchProducerCUDA):

        if 'hltSiPixelDigisLegacy' not in process.__dict__:
            # reconstruct the pixel digis on the cpu
            process.hltSiPixelDigisLegacy = process.hltSiPixelDigis.clone()

        # SwitchProducer wrapping a subset of the legacy pixel digis producer, or the conversion of the pixel digis errors to the legacy format
        process.hltSiPixelDigis = SwitchProducerCUDA(
            # legacy producer
            cpu = cms.EDAlias(
                hltSiPixelDigisLegacy = cms.VPSet(
                    cms.PSet(type = cms.string("DetIdedmEDCollection")),
                    cms.PSet(type = cms.string("SiPixelRawDataErroredmDetSetVector")),
                    cms.PSet(type = cms.string("PixelFEDChanneledmNewDetSetVector"))
                )
            )
        )

    elif not hasattr(process.hltSiPixelDigis, 'cpu'):
        raise Exception('unsupported configuration: "process.hltSiPixelDigis" is a SwitchProducerCUDA, but does not have a "cpu" branch')

    # conversion from SoA to legacy format
    _clone_if_missing(process.hltSiPixelDigis, 'cuda', 'EventFilter.SiPixelRawToDigi.siPixelDigiErrorsFromSoA_cfi', 'siPixelDigiErrorsFromSoA',
        digiErrorSoASrc = "hltSiPixelDigiErrorsSoA",
        UsePhase1 = True
    )


    # SwitchProducer: hltSiPixelClusters
    if not isinstance(process.hltSiPixelClusters, SwitchProducerCUDA):

        if 'hltSiPixelClustersLegacy' not in process.__dict__:
            # reconstruct the pixel clusters on the cpu
            process.hltSiPixelClustersLegacy = process.hltSiPixelClusters.clone(
                src = "hltSiPixelDigisLegacy"
            )

        # SwitchProducer wrapping a subset of the legacy pixel cluster producer, or the conversion of the pixel digis (except errors) and clusters to the legacy format
        process.hltSiPixelClusters = SwitchProducerCUDA(
            # legacy producer
            cpu = cms.EDAlias(
                hltSiPixelClustersLegacy = cms.VPSet(
                    cms.PSet(type = cms.string("SiPixelClusteredmNewDetSetVector"))
                )
            )
        )

    elif not hasattr(process.hltSiPixelClusters, 'cpu'):
        raise Exception('unsupported configuration: "process.hltSiPixelClusters" is a SwitchProducerCUDA, but does not have a "cpu" branch')

    # conversion from SoA to legacy format
    _clone_if_missing(process.hltSiPixelClusters, 'cuda', 'RecoLocalTracker.SiPixelClusterizer.siPixelDigisClustersFromSoA_cfi', 'siPixelDigisClustersFromSoA',
        src = 'hltSiPixelDigisSoA',
        produceDigis = False,
        storeDigis = False,
        # use the same thresholds as the legacy module
        clusterThreshold_layer1 = process.hltSiPixelClustersLegacy.ClusterThreshold_L1,
        clusterThreshold_otherLayers = process.hltSiPixelClustersLegacy.ClusterThreshold
    )

    # reconstruct the pixel rechits on the gpu
    _clone_if_missing(process, 'hltSiPixelRecHitsCUDA', 'RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDA_cfi', 'siPixelRecHitCUDA',
        src = 'hltSiPixelClustersCUDA',
        beamSpot = 'hltOnlineBeamSpotToCUDA'
    )

    # cpu only: produce the pixel rechits in SoA and legacy format, from the legacy clusters
    _clone_if_missing(process, 'hltSiPixelRecHitSoA', 'RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacy_cfi', 'siPixelRecHitSoAFromLegacy',
        src = 'hltSiPixelClusters',
        beamSpot = 'hltOnlineBeamSpot',
        convertToLegacy = True
    )

    # SwitchProducer: hltSiPixelRecHits
    if not isinstance(process.hltSiPixelRecHits, SwitchProducerCUDA):
        # SwitchProducer wrapping the legacy pixel rechit producer or the transfer of the pixel rechits to the host and the conversion from SoA
        process.hltSiPixelRecHits = SwitchProducerCUDA(
            # legacy producer
            cpu = cms.EDAlias(
                hltSiPixelRecHitSoA = cms.VPSet(
                    cms.PSet(type = cms.string("SiPixelRecHitedmNewDetSetVector")),
                    cms.PSet(type = cms.string("uintAsHostProduct"))
                )
            )
        )

    elif not hasattr(process.hltSiPixelRecHits, 'cpu'):
        raise Exception('unsupported configuration: "process.hltSiPixelRecHits" is a SwitchProducerCUDA, but does not have a "cpu" branch')

    # conversion from SoA to legacy format
    _clone_if_missing(process.hltSiPixelRecHits, 'cuda', 'RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDA_cfi', 'siPixelRecHitFromCUDA',
        pixelRecHitSrc = 'hltSiPixelRecHitsCUDA',
        src = 'hltSiPixelClusters'
    )


    # Tasks and Sequences

    if 'HLTDoLocalPixelTask' not in process.__dict__:
        process.HLTDoLocalPixelTask = cms.Task(
            process.hltOnlineBeamSpotToCUDA,   # transfer the beamspot to the gpu
            process.hltSiPixelClustersCUDA,    # reconstruct the pixel digis and clusters on the gpu
            process.hltSiPixelRecHitsCUDA,     # reconstruct the pixel rechits on the gpu
            process.hltSiPixelDigisSoA,        # copy the pixel digis (except errors) and clusters to the host
            process.hltSiPixelDigiErrorsSoA,   # copy the pixel digis errors to the host
            process.hltSiPixelDigisLegacy,     # legacy pixel digis producer
            process.hltSiPixelDigis,           # SwitchProducer wrapping a subset of the legacy pixel digis producer, or the conversion of the pixel digis errors from SoA
            process.hltSiPixelClustersLegacy,  # legacy pixel cluster producer
            process.hltSiPixelClusters,        # SwitchProducer wrapping a subset of the legacy pixel cluster producer, or the conversion of the pixel digis (except errors) and clusters from SoA
            process.hltSiPixelClustersCache,   # legacy module, used by the legacy pixel quadruplet producer
            process.hltSiPixelRecHitSoA,       # pixel rechits on cpu, in SoA & legacy format
            process.hltSiPixelRecHits,         # SwitchProducer wrapping the legacy pixel rechit producer or the transfer of the pixel rechits to the host and the conversion from SoA
        )

    elif not isinstance(process.HLTDoLocalPixelTask, cms.Task):
        raise Exception('unsupported configuration: "process.HLTDoLocalPixelTask" already exists, but it is not a Task')

    # redefine HLTDoLocalPixelSequence (it was emptied at the start of this function)
    process.HLTDoLocalPixelSequence = cms.Sequence(process.HLTDoLocalPixelTask)


    # workaround for old version of AlCa_LumiPixelsCounts paths
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

    # SwitchProducer: hltPixelTracksSoA
    if not ('hltPixelTracksSoA' in process.__dict__ and isinstance(process.hltPixelTracksSoA, SwitchProducerCUDA)):
        # build pixel ntuplets and pixel tracks in SoA format on gpu
        _clone(process, 'hltPixelTracksCUDA', 'RecoPixelVertexing.PixelTriplets.pixelTracksCUDA_cfi', 'pixelTracksCUDA',
            pixelRecHitSrc = 'hltSiPixelRecHitsCUDA',
            onGPU = True,
            idealConditions = False
        )
        # use quality cuts tuned for Run-2 ideal conditions for all Run-3 workflows
        run3_common.toModify(process.hltPixelTracksCUDA, idealConditions = True)

        process.hltPixelTracksSoA = SwitchProducerCUDA(
            # build pixel ntuplets and pixel tracks in SoA format on cpu
            cpu = process.hltPixelTracksCUDA.clone(
                pixelRecHitSrc = 'hltSiPixelRecHitSoA',
                onGPU = False
            )
        )

    elif hasattr(process.hltPixelTracksSoA, 'cpu'):
        # if cpu branch of SwitchProducerCUDA exists, take hltPixelTracksCUDA (gpu)
        # from hltPixelTracksSoA.cpu (cpu) to enforce same configuration parameters
        process.hltPixelTracksCUDA = process.hltPixelTracksSoA.cpu.clone(
            pixelRecHitSrc = "hltSiPixelRecHitsCUDA",
            onGPU = True
        )

    else:
        raise Exception('unsupported configuration: "process.hltPixelTracksSoA" is a SwitchProducerCUDA, but does not have a "cpu" branch')

    # transfer the pixel tracks in SoA format to cpu
    _clone_if_missing(process.hltPixelTracksSoA, 'cuda', 'RecoPixelVertexing.PixelTrackFitting.pixelTracksSoA_cfi', 'pixelTracksSoA',
        src = 'hltPixelTracksCUDA'
    )


    # convert the pixel tracks from SoA to legacy format
    if process.hltPixelTracks.type_() != 'PixelTrackProducerFromSoA':
        _clone(process, 'hltPixelTracks', 'RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi', 'pixelTrackProducerFromSoA',
            beamSpot = 'hltOnlineBeamSpot',
            pixelRecHitLegacySrc = 'hltSiPixelRecHits',
            trackSrc = 'hltPixelTracksSoA'
        )


    # referenced in process.HLTRecopixelvertexingTask
    if hasHLTPixelVertexReco:

        # SwitchProducer: hltPixelVerticesSoA
        if not ('hltPixelVerticesSoA' in process.__dict__ and isinstance(process.hltPixelVerticesSoA, SwitchProducerCUDA)):
            # build pixel vertices in SoA format on gpu
            _clone(process, 'hltPixelVerticesCUDA', 'RecoPixelVertexing.PixelVertexFinding.pixelVerticesCUDA_cfi', 'pixelVerticesCUDA',
                pixelTrackSrc = 'hltPixelTracksCUDA',
                onGPU = True
            )

            # build or transfer pixel vertices in SoA format on cpu
            process.hltPixelVerticesSoA = SwitchProducerCUDA(
                # build pixel vertices in SoA format on cpu
                cpu = process.hltPixelVerticesCUDA.clone(
                    pixelTrackSrc = 'hltPixelTracksSoA',
                    onGPU = False
                )
            )

        elif hasattr(process.hltPixelVerticesSoA, 'cpu'):
            # if cpu branch of SwitchProducerCUDA exists, take hltPixelVerticesCUDA (gpu)
            # from hltPixelVerticesSoA.cpu (cpu) to enforce same configuration parameters
            process.hltPixelVerticesCUDA = process.hltPixelVerticesSoA.cpu.clone(
                pixelTrackSrc = 'hltPixelTracksCUDA',
                onGPU = True
            )

        else:
            raise Exception('unsupported configuration: "process.hltPixelVerticesSoA" is a SwitchProducerCUDA, but does not have a "cpu" branch')

        # transfer the pixel vertices in SoA format to cpu
        _clone_if_missing(process.hltPixelVerticesSoA, 'cuda', 'RecoPixelVertexing.PixelVertexFinding.pixelVerticesSoA_cfi', 'pixelVerticesSoA',
            src = 'hltPixelVerticesCUDA'
        )


        # convert the pixel vertices from SoA to legacy format
        if process.hltPixelVertices.type_() != 'PixelVertexProducerFromSoA':
            _clone(process, 'hltPixelVertices', 'RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi', 'pixelVertexFromSoA',
                src = 'hltPixelVerticesSoA',
                TrackCollection = 'hltPixelTracks',
                beamSpot = 'hltOnlineBeamSpot'
            )


    # Tasks and Sequences

    if 'HLTRecoPixelTracksTask' not in process.__dict__:
        process.HLTRecoPixelTracksTask = cms.Task(
            process.hltPixelTracksTrackingRegions,         # from the original sequence
            process.hltPixelTracksCUDA,                    # pixel ntuplets on gpu, in SoA format
            process.hltPixelTracksSoA,                     # pixel ntuplets on cpu, in SoA format
            process.hltPixelTracks,                        # pixel tracks on cpu, in legacy format
        )

    elif not isinstance(process.HLTRecoPixelTracksTask, cms.Task):
        raise Exception('unsupported configuration: "process.HLTRecoPixelTracksTask" already exists, but it is not a Task')

    # redefine HLTRecoPixelTracksSequence (it was emptied at the start of this function)
    process.HLTRecoPixelTracksSequence = cms.Sequence(process.HLTRecoPixelTracksTask)

    if hasHLTPixelVertexReco:

        if 'HLTRecopixelvertexingTask' not in process.__dict__:
            process.HLTRecopixelvertexingTask = cms.Task(
                process.HLTRecoPixelTracksTask,
                process.hltPixelVerticesCUDA,              # pixel vertices on gpu, in SoA format
                process.hltPixelVerticesSoA,               # pixel vertices on cpu, in SoA format
                process.hltPixelVertices,                  # pixel vertices on cpu, in legacy format
                process.hltTrimmedPixelVertices,           # from the original sequence
            )

        elif not isinstance(process.HLTRecopixelvertexingTask, cms.Task):
            raise Exception('unsupported configuration: "process.HLTRecopixelvertexingTask" already exists, but it is not a Task')

        # redefine HLTRecopixelvertexingSequence (it was emptied at the start of this function)
        process.HLTRecopixelvertexingSequence = cms.Sequence(
            process.hltPixelTracksFitter +             # not used here, kept for compatibility with legacy sequences
            process.hltPixelTracksFilter,              # not used here, kept for compatibility with legacy sequences
            process.HLTRecopixelvertexingTask,
        )


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

    _load_if_missing(process, 'ecalElectronicsMappingGPUESProducer', 'EventFilter.EcalRawToDigi.ecalElectronicsMappingGPUESProducer_cfi')
    _load_if_missing(process, 'ecalGainRatiosGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalGainRatiosGPUESProducer_cfi')
    _load_if_missing(process, 'ecalPedestalsGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalPedestalsGPUESProducer_cfi')
    _load_if_missing(process, 'ecalPulseCovariancesGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalPulseCovariancesGPUESProducer_cfi')
    _load_if_missing(process, 'ecalPulseShapesGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalPulseShapesGPUESProducer_cfi')
    _load_if_missing(process, 'ecalSamplesCorrelationGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalSamplesCorrelationGPUESProducer_cfi')
    _load_if_missing(process, 'ecalTimeBiasCorrectionsGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalTimeBiasCorrectionsGPUESProducer_cfi')
    _load_if_missing(process, 'ecalTimeCalibConstantsGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalTimeCalibConstantsGPUESProducer_cfi')
    _load_if_missing(process, 'ecalMultifitParametersGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalMultifitParametersGPUESProducer_cfi')
    _load_if_missing(process, 'ecalRechitADCToGeVConstantGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalRechitADCToGeVConstantGPUESProducer_cfi')
    _load_if_missing(process, 'ecalRechitChannelStatusGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalRechitChannelStatusGPUESProducer_cfi')
    _load_if_missing(process, 'ecalIntercalibConstantsGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalIntercalibConstantsGPUESProducer_cfi')
    _load_if_missing(process, 'ecalLaserAPDPNRatiosGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalLaserAPDPNRatiosGPUESProducer_cfi')
    _load_if_missing(process, 'ecalLaserAPDPNRatiosRefGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalLaserAPDPNRatiosRefGPUESProducer_cfi')
    _load_if_missing(process, 'ecalLaserAlphasGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalLaserAlphasGPUESProducer_cfi')
    _load_if_missing(process, 'ecalLinearCorrectionsGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalLinearCorrectionsGPUESProducer_cfi')
    _load_if_missing(process, 'ecalRecHitParametersGPUESProducer', 'RecoLocalCalo.EcalRecProducers.ecalRecHitParametersGPUESProducer_cfi')


    # Modules and EDAliases

    # ECAL unpacker running on gpu
    _clone_if_missing(process, 'hltEcalDigisGPU', 'EventFilter.EcalRawToDigi.ecalRawToDigiGPU_cfi', 'ecalRawToDigiGPU')

    # SwitchProducer: hltEcalDigis
    if not isinstance(process.hltEcalDigis, SwitchProducerCUDA):

        if 'hltEcalDigisLegacy' not in process.__dict__:
            process.hltEcalDigisLegacy = process.hltEcalDigis.clone()
        else:
            raise Exception('unsupported configuration: "process.hltEcalDigis" is not a SwitchProducerCUDA, but "process.hltEcalDigisLegacy" already exists')

        # SwitchProducer wrapping the legacy ECAL unpacker or the ECAL digi converter from SoA format on gpu to legacy format on cpu
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
            )
        )

    # convert ECAL digis from SoA format on gpu to legacy format on cpu
    _clone_if_missing(process.hltEcalDigis, 'cuda', 'EventFilter.EcalRawToDigi.ecalCPUDigisProducer_cfi', 'ecalCPUDigisProducer',
        digisInLabelEB = ('hltEcalDigisGPU', 'ebDigis'),
        digisInLabelEE = ('hltEcalDigisGPU', 'eeDigis'),
        produceDummyIntegrityCollections = True
    )

    # ECAL multifit running on gpu
    _clone_if_missing(process, 'hltEcalUncalibRecHitGPU', 'RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitProducerGPU_cfi', 'ecalUncalibRecHitProducerGPU',
        digisLabelEB = ('hltEcalDigisGPU', 'ebDigis'),
        digisLabelEE = ('hltEcalDigisGPU', 'eeDigis'),
        shouldRunTimingComputation = False
    )

    # copy the ECAL uncalibrated rechits from gpu to cpu in SoA format
    _clone_if_missing(process, 'hltEcalUncalibRecHitSoA', 'RecoLocalCalo.EcalRecProducers.ecalCPUUncalibRecHitProducer_cfi', 'ecalCPUUncalibRecHitProducer',
        recHitsInLabelEB = ('hltEcalUncalibRecHitGPU', 'EcalUncalibRecHitsEB'),
        recHitsInLabelEE = ('hltEcalUncalibRecHitGPU', 'EcalUncalibRecHitsEE'),
    )

    # SwitchProducer: hltEcalUncalibRecHit
    if not isinstance(process.hltEcalUncalibRecHit, SwitchProducerCUDA):
        # SwitchProducer wrapping the legacy ECAL uncalibrated rechits producer or a converter from SoA to legacy format
        process.hltEcalUncalibRecHit = SwitchProducerCUDA(
            # legacy producer
            cpu = process.hltEcalUncalibRecHit
        )

    # convert the ECAL uncalibrated rechits from SoA to legacy format
    _clone_if_missing(process.hltEcalUncalibRecHit, 'cuda', 'RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitConvertGPU2CPUFormat_cfi', 'ecalUncalibRecHitConvertGPU2CPUFormat',
        recHitsLabelGPUEB = ('hltEcalUncalibRecHitSoA', 'EcalUncalibRecHitsEB'),
        recHitsLabelGPUEE = ('hltEcalUncalibRecHitSoA', 'EcalUncalibRecHitsEE'),
    )

    # Reconstructing the ECAL calibrated rechits on gpu works, but is extremely slow.
    # Disable it for the time being, until the performance has been addressed.
    """
    _clone_if_missing(process, 'hltEcalRecHitGPU', 'RecoLocalCalo.EcalRecProducers.ecalRecHitGPU_cfi', 'ecalRecHitGPU',
        uncalibrecHitsInLabelEB = ("hltEcalUncalibRecHitGPU","EcalUncalibRecHitsEB"),
        uncalibrecHitsInLabelEE = ("hltEcalUncalibRecHitGPU","EcalUncalibRecHitsEE"),
    )

    _clone_if_missing(process, 'hltEcalRecHitSoA', 'RecoLocalCalo.EcalRecProducers.ecalCPURecHitProducer_cfi', 'ecalCPURecHitProducer',
        recHitsInLabelEB = ("hltEcalRecHitGPU", "EcalRecHitsEB"),
        recHitsInLabelEE = ("hltEcalRecHitGPU", "EcalRecHitsEE"),
    )

    # SwitchProducer wrapping the legacy ECAL calibrated rechits producer or a converter from SoA to legacy format
    if not isinstance(process.hltEcalRecHit, SwitchProducerCUDA):
        process.hltEcalRecHit = SwitchProducerCUDA(
            # legacy producer
            cpu = process.hltEcalRecHit,
        )

    # convert the ECAL calibrated rechits from SoA to legacy format
    _clone_if_missing(process.hltEcalRecHit, 'cuda', 'RecoLocalCalo.EcalRecProducers.ecalRecHitConvertGPU2CPUFormat_cfi', 'ecalRecHitConvertGPU2CPUFormat',
        recHitsLabelGPUEB = ("hltEcalRecHitSoA", "EcalRecHitsEB"),
        recHitsLabelGPUEE = ("hltEcalRecHitSoA", "EcalRecHitsEE"),
    )
    """

    # SwitchProducer wrapping the legacy ECAL rechits producer
    # the gpu unpacker does not produce the TPs used for the recovery, so the SwitchProducer alias does not provide them:
    #   - the cpu uncalibrated rechit producer may mark them for recovery, read the TPs explicitly from the legacy unpacker
    #   - the gpu uncalibrated rechit producer does not flag them for recovery, so the TPs are not necessary
    if not isinstance(process.hltEcalRecHit, SwitchProducerCUDA):
        process.hltEcalRecHit = SwitchProducerCUDA(
            cpu = process.hltEcalRecHit.clone(
                triggerPrimitiveDigiCollection = ('hltEcalDigisLegacy', 'EcalTriggerPrimitives')
            )
        )

    elif not hasattr(process.hltEcalRecHit, 'cpu'):
        raise Exception('unsupported configuration: "process.hltEcalRecHit" is a SwitchProducerCUDA, but does not have a "cpu" branch')

    if not hasattr(process.hltEcalRecHit, 'cuda'):
        process.hltEcalRecHit.cuda = process.hltEcalRecHit.cpu.clone(
            triggerPrimitiveDigiCollection = 'unused'
        )


    # enforce consistent configuration of CPU and GPU modules for timing of ECAL RecHits
    if process.hltEcalUncalibRecHit.cpu.algoPSet.timealgo == 'RatioMethod':
        process.hltEcalUncalibRecHitGPU.shouldRunTimingComputation = True
        process.hltEcalUncalibRecHitSoA.containsTimingInformation = True
        for _parName in [
            'EBtimeFitLimits_Lower',
            'EBtimeFitLimits_Upper',
            'EEtimeFitLimits_Lower',
            'EEtimeFitLimits_Upper',
            'EBtimeConstantTerm',
            'EEtimeConstantTerm',
            'EBtimeNconst',
            'EEtimeNconst',
            'outOfTimeThresholdGain12pEB',
            'outOfTimeThresholdGain12pEE',
            'outOfTimeThresholdGain12mEB',
            'outOfTimeThresholdGain12mEE',
            'outOfTimeThresholdGain61pEB',
            'outOfTimeThresholdGain61pEE',
            'outOfTimeThresholdGain61mEB',
            'outOfTimeThresholdGain61mEE',
        ]:
            setattr(process.hltEcalUncalibRecHitGPU, _parName, getattr(process.hltEcalUncalibRecHit.cpu.algoPSet, _parName))
    # note: the "RatioMethod" is the only one available in the GPU implementation
    elif process.hltEcalUncalibRecHit.cpu.algoPSet.timealgo != 'None':
        _logMsg = '"process.hltEcalUncalibRecHit.cpu.algoPSet.timealgo = \''+process.hltEcalUncalibRecHit.cpu.algoPSet.timealgo+'\'"'
        _logMsg += ' has no counterpart in the GPU implementation of the ECAL local reconstruction (use "None" or "RatioMethod")'
        raise Exception('unsupported configuration: '+_logMsg)


    # Tasks and Sequences

    if 'HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask' not in process.__dict__:
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask = cms.Task(
            process.hltEcalDigisGPU,             # unpack ECAL digis on gpu
            process.hltEcalDigisLegacy,          # legacy producer, referenced in the SwitchProducer
            process.hltEcalDigis,                # SwitchProducer
            process.hltEcalUncalibRecHitGPU,     # run ECAL local reconstruction and multifit on gpu
            process.hltEcalUncalibRecHitSoA,     # needed by hltEcalPhiSymFilter - copy to host
            process.hltEcalUncalibRecHit,        # needed by hltEcalPhiSymFilter - convert to legacy format
#           process.hltEcalRecHitGPU,            # make ECAL calibrated rechits on gpu
#           process.hltEcalRecHitSoA,            # copy to host
            process.hltEcalDetIdToBeRecovered,   # legacy producer
            process.hltEcalRecHit,               # legacy producer
        )

    elif not isinstance(process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask, cms.Task):
        raise Exception('unsupported configuration: "process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask" already exists, but it is not a Task')

    # redefine HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence (it was emptied at the start of this function)
    process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence = cms.Sequence(
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask
    )

    if hasHLTEcalPreshowerSeq:

        if 'HLTPreshowerTask' not in process.__dict__:
            process.HLTPreshowerTask = cms.Task(
                process.hltEcalPreshowerDigis,   # unpack ECAL preshower digis on the host
                process.hltEcalPreshowerRecHit,  # build ECAL preshower rechits on the host
            )

        elif not isinstance(process.HLTPreshowerTask, cms.Task):
            raise Exception('unsupported configuration: "process.HLTPreshowerTask" already exists, but it is not a Task')

        # redefine HLTPreshowerSequence (it was emptied at the start of this function)
        process.HLTPreshowerSequence = cms.Sequence(process.HLTPreshowerTask)

        if 'HLTDoFullUnpackingEgammaEcalTask' not in process.__dict__:
            process.HLTDoFullUnpackingEgammaEcalTask = cms.Task(
                process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask,
                process.HLTPreshowerTask,
            )

        elif not isinstance(process.HLTDoFullUnpackingEgammaEcalTask, cms.Task):
            raise Exception('unsupported configuration: "process.HLTDoFullUnpackingEgammaEcalTask" already exists, but it is not a Task')

        # redefine sequences (they were emptied at the start of this function)
        process.HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence(process.HLTDoFullUnpackingEgammaEcalTask)
        process.HLTDoFullUnpackingEgammaEcalMFSequence = cms.Sequence(process.HLTDoFullUnpackingEgammaEcalTask)


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

    _load_if_missing(process, 'hcalElectronicsMappingGPUESProducer', 'EventFilter.HcalRawToDigi.hcalElectronicsMappingGPUESProducer_cfi')
    _load_if_missing(process, 'hcalChannelQualityGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalChannelQualityGPUESProducer_cfi')
    _load_if_missing(process, 'hcalGainsGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalGainsGPUESProducer_cfi')
    _load_if_missing(process, 'hcalGainWidthsGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalGainWidthsGPUESProducer_cfi')
    _load_if_missing(process, 'hcalLUTCorrsGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalLUTCorrsGPUESProducer_cfi')
    _load_if_missing(process, 'hcalConvertedPedestalsGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalConvertedPedestalsGPUESProducer_cfi')
    _load_if_missing(process, 'hcalConvertedPedestalWidthsGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalConvertedPedestalWidthsGPUESProducer_cfi')
    _load_if_missing(process, 'hcalQIECodersGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalQIECodersGPUESProducer_cfi')
    _load_if_missing(process, 'hcalRecoParamsWithPulseShapesGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalRecoParamsWithPulseShapesGPUESProducer_cfi')
    _load_if_missing(process, 'hcalRespCorrsGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalRespCorrsGPUESProducer_cfi')
    _load_if_missing(process, 'hcalTimeCorrsGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalTimeCorrsGPUESProducer_cfi')
    _load_if_missing(process, 'hcalQIETypesGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalQIETypesGPUESProducer_cfi')
    _load_if_missing(process, 'hcalSiPMParametersGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalSiPMParametersGPUESProducer_cfi')
    _load_if_missing(process, 'hcalSiPMCharacteristicsGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalSiPMCharacteristicsGPUESProducer_cfi')
    _load_if_missing(process, 'hcalMahiPulseOffsetsGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalMahiPulseOffsetsGPUESProducer_cfi')

    _clone_if_missing(process, 'hcalConvertedEffectivePedestalsGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalConvertedEffectivePedestalsGPUESProducer_cfi',
        label0 = 'withTopoEff'
    )

    _clone_if_missing(process, 'hcalConvertedEffectivePedestalWidthsGPUESProducer', 'RecoLocalCalo.HcalRecProducers.hcalConvertedEffectivePedestalWidthsGPUESProducer_cfi',
        label0 = 'withTopoEff',
        label1 = 'withTopoEff'
    )


    # Modules and EDAliases

    # The HCAL unpacker running on the gpu supports only the HB and HE digis.
    # So, run the legacy unacker on the cpu, then convert the HB and HE digis
    # to SoA format and copy them to the gpu.

    _clone_if_missing(process, 'hltHcalDigisGPU', 'EventFilter.HcalRawToDigi.hcalDigisProducerGPU_cfi', 'hcalDigisProducerGPU',
        hbheDigisLabel = 'hltHcalDigis',
        qie11DigiLabel = 'hltHcalDigis',
        digisLabelF01HE = '',
        digisLabelF5HB = '',
        digisLabelF3HB = ''
    )

    # run the HCAL local reconstruction (including Method 0 and MAHI) on gpu
    _clone_if_missing(process, 'hltHbherecoGPU', 'RecoLocalCalo.HcalRecProducers.hbheRecHitProducerGPU_cfi', 'hbheRecHitProducerGPU',
        digisLabelF01HE = 'hltHcalDigisGPU',
        digisLabelF5HB = 'hltHcalDigisGPU',
        digisLabelF3HB = 'hltHcalDigisGPU',
        recHitsLabelM0HBHE = ''
    )

    # transfer the HCAL rechits to the cpu, and convert them to the legacy format
    _clone_if_missing(process, 'hltHbherecoFromGPU', 'RecoLocalCalo.HcalRecProducers.hcalCPURecHitsProducer_cfi', 'hcalCPURecHitsProducer',
        recHitsM0LabelIn = 'hltHbherecoGPU',
        recHitsM0LabelOut = '',
        recHitsLegacyLabelOut = ''
    )

    # SwitchProducer between the legacy producer and the copy from gpu with conversion
    if not isinstance(process.hltHbhereco, SwitchProducerCUDA):
        process.hltHbhereco = SwitchProducerCUDA(
            # legacy producer
            cpu = process.hltHbhereco.clone()
        )

    elif not hasattr(process.hltHbhereco, 'cpu'):
        raise Exception('unsupported configuration: "process.hltHbhereco" is a SwitchProducerCUDA, but does not have a "cpu" branch')

    if not hasattr(process.hltHbhereco, 'cuda'):
        # alias to the rechits converted to legacy format
        process.hltHbhereco.cuda = cms.EDAlias(
            hltHbherecoFromGPU = cms.VPSet(
                cms.PSet(type = cms.string('HBHERecHitsSorted'))
            )
        )


    # Tasks and Sequences

    if hasHLTDoLocalHcalSeq:
        if 'HLTDoLocalHcalTask' not in process.__dict__:
            process.HLTDoLocalHcalTask = cms.Task(
                process.hltHcalDigis,       # legacy producer, unpack HCAL digis on cpu
                process.hltHcalDigisGPU,    # copy to gpu and convert to SoA format
                process.hltHbherecoGPU,     # run the HCAL local reconstruction (including Method 0 and MAHI) on gpu
                process.hltHbherecoFromGPU, # transfer the HCAL rechits to the cpu, and convert them to the legacy format
                process.hltHbhereco,        # SwitchProducer between the legacy producer and the copy from gpu with conversion
                process.hltHfprereco,       # legacy producer
                process.hltHfreco,          # legacy producer
                process.hltHoreco           # legacy producer
            )

        elif not isinstance(process.HLTDoLocalHcalTask, cms.Task):
            raise Exception('unsupported configuration: "process.HLTDoLocalHcalTask" already exists, but it is not a Task')

        # redefine HLTDoLocalHcalSequence (it was emptied at the start of this function)
        process.HLTDoLocalHcalSequence = cms.Sequence(process.HLTDoLocalHcalTask)

    if 'HLTStoppedHSCPLocalHcalRecoTask' not in process.__dict__:
        process.HLTStoppedHSCPLocalHcalRecoTask = cms.Task(
            process.hltHcalDigis,           # legacy producer, unpack HCAL digis on cpu
            process.hltHcalDigisGPU,        # copy to gpu and convert to SoA format
            process.hltHbherecoGPU,         # run the HCAL local reconstruction (including Method 0 and MAHI) on gpu
            process.hltHbherecoFromGPU,     # transfer the HCAL rechits to the cpu, and convert them to the legacy format
            process.hltHbhereco             # SwitchProducer between the legacy producer and the copy from gpu with conversion
        )

    elif not isinstance(process.HLTStoppedHSCPLocalHcalRecoTask, cms.Task):
        raise Exception('unsupported configuration: "process.HLTStoppedHSCPLocalHcalRecoTask" already exists, but it is not a Task')

    # redefine HLTStoppedHSCPLocalHcalReco (it was emptied at the start of this function)
    process.HLTStoppedHSCPLocalHcalReco = cms.Sequence(process.HLTStoppedHSCPLocalHcalRecoTask)


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
