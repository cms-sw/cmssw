import FWCore.ParameterSet.Config as cms
import os

def setupLocalInputsForRelVal(local_daq : str = 'local_daq'):

    # Setup DTH source
    inputdir = '/store/group/dpg_hgcal/comm_hgcal/relval'
    files = [
        'run20000000_ls0000_EoR_source2000.jsn',
        'run20000000_ls0000_EoR_source2001.jsn',
        'run20000000_ls0001_EoLS_source2000.jsn',
        'run20000000_ls0001_EoLS_source2001.jsn',
        'run20000000_ls0001_index000000_source2000.raw',
        'run20000000_ls0001_index000000_source2001.raw',
    ]
    localdir = f'{local_daq}/ramdisk/run20000000'
    os.makedirs(localdir, exist_ok=True)
    for f in files:
        os.system(f'xrdcp --silent -f root://cms-xrd-global.cern.ch//{inputdir}/{f} {localdir}')


def setupConditionsForSep2024TB(process):

    process.load('Configuration.StandardSequences.Accelerators_cff')

    process.load(f"Configuration.Geometry.GeometryExtendedRun4D104Reco_cff")
    process.load(f"Configuration.Geometry.GeometryExtendedRun4D104_cff")
    from Geometry.HGCalMapping.hgcalmapping_cff import customise_hgcalmapper
    process = customise_hgcalmapper(
        process, modules='Geometry/HGCalMapping/data/ModuleMaps/modulelocator_P5v1.txt')

    process.hgcalConfigESProducer = cms.ESSource(
        "HGCalConfigurationESProducer", bePassthroughMode=cms.int32(-1),
        cbHeaderMarker=cms.int32(-1),
        charMode=cms.int32(-1),
        econdHeaderMarker=cms.int32(-1),
        fedjson=cms.FileInPath('RecoLocalCalo/HGCalRecProducers/data/testbeam/config_feds_v1.json'),
        indexSource=cms.ESInputTag("hgCalMappingESProducer", ""),
        modjson=cms.FileInPath('RecoLocalCalo/HGCalRecProducers/data/testbeam//config_econds_v1.json'),
        slinkHeaderMarker=cms.int32(-1))

    return process


def runRecoForSep2024TB(process):

    local_daq = setupLocalInputsForRelVal()
    
    process = setupConditionsForSep2024TB(process)

    process.EvFDaqDirector = cms.Service("EvFDaqDirector",
                                         baseDir=cms.untracked.string('local_daq/fu'),
                                         buBaseDir=cms.untracked.string('local_daq/ramdisk'),
                                         buBaseDirsAll=cms.untracked.vstring('local_daq/ramdisk'),
                                         buBaseDirsNumStreams=cms.untracked.vint32(2),
                                         buBaseDirsStreamIDs=cms.untracked.vint32(2000, 2001),
                                         directorIsBU=cms.untracked.bool(False),
                                         fileBrokerHost=cms.untracked.string('htcp40.cern.ch'),
                                         fileBrokerHostFromCfg=cms.untracked.bool(False),
                                         runNumber=cms.untracked.uint32(20000000),
                                         sourceIdentifier=cms.untracked.string('source'),
                                         useFileBroker=cms.untracked.bool(True)
                                         )

    process.source = cms.Source("DAQSource",
                                dataMode=cms.untracked.string('DTH'),
                                eventChunkBlock=cms.untracked.uint32(100),
                                eventChunkSize=cms.untracked.uint32(100),
                                fileDiscoveryMode=cms.untracked.bool(True),
                                keepRawFiles=cms.untracked.bool(True),
                                maxBufferedFiles=cms.untracked.uint32(2),
                                maxChunkSize=cms.untracked.uint32(1000),
                                numBuffers=cms.untracked.uint32(3),
                                testing=cms.untracked.bool(True),
                                useL1EventID=cms.untracked.bool(False),
                                verifyChecksum=cms.untracked.bool(False)
                                )

    process.FastMonitoringService = cms.Service("FastMonitoringService",
                                                sleepTime=cms.untracked.int32(1)
                                                )

    # Setup HGCal unpacker
    process.hgcalDigis = cms.EDProducer("HGCalRawToDigi",
                                        src=cms.InputTag("rawDataCollector")
                                        )

    # ESProducer to load calibration parameters from JSON file, like pedestals
    process.hgcalCalibParamESProducer = cms.ESProducer(
        'hgcalrechit::HGCalCalibrationESProducer@alpaka',
        filename=cms.FileInPath('RecoLocalCalo/HGCalRecProducers/data/testbeam/level0_calib_params_test.json'),
        filenameEnergyLoss=cms.FileInPath('RecoLocalCalo/HGCalRecProducers/data/testbeam/hgcal_energyloss_v16.json'),
        indexSource=cms.ESInputTag('hgCalMappingESProducer', ''),
        mapSource=cms.ESInputTag('hgCalMappingModuleESProducer', ''))

    process.hgcalSoARecHits = cms.EDProducer('HGCalRecHitsProducer@alpaka',
                                             digis=cms.InputTag('hgcalDigis', ''),
                                             calibSource=cms.ESInputTag('hgcalCalibParamESProducer', ''),
                                             n_hits_scale=cms.int32(1),
                                             n_blocks=cms.int32(1024),
                                             n_threads=cms.int32(1024),
                                             k_noise=cms.double(5.)
                                             )

    from RecoTICL.LayerClustering.hgCalSoARecHitsLayerClustersProducer_cfi import hgCalSoARecHitsLayerClustersProducer
    process.hgcalSoARecHitsLayerClusters = hgCalSoARecHitsLayerClustersProducer.clone(
        hgcalRecHitsSoA="hgcalSoARecHits"
    )

    from RecoTICL.LayerClustering.hgCalSoALayerClustersProducer_cfi import hgCalSoALayerClustersProducer
    process.hgcalSoALayerClusters = hgCalSoALayerClustersProducer.clone(
        hgcalRecHitsLayerClustersSoA="hgcalSoARecHitsLayerClusters",
        hgcalRecHitsSoA="hgcalSoARecHits"
    )

    from RecoTICL.LayerClustering.hgCalLayerClustersFromSoAProducer_cfi import hgCalLayerClustersFromSoAProducer
    process.hgcalMergeLayerClusters = hgCalLayerClustersFromSoAProducer.clone(
        hgcalRecHitsLayerClustersSoA="hgcalSoARecHitsLayerClusters",
        hgcalRecHitsSoA="hgcalSoARecHits",
        src="hgcalSoALayerClusters"
    )

    process.reco_task = cms.Task(
        process.hgcalDigis,
        process.hgcalSoARecHits,
        process.hgcalSoARecHitsLayerClusters,
        process.hgcalSoALayerClusters,
        process.hgcalMergeLayerClusters
    )
    process.hgcalTestBeamLocalRecoSequence = cms.Path(process.reco_task)
    process.schedule.insert(0, process.hgcalTestBeamLocalRecoSequence)

    # HGCal DQM clients, saved as DQMIO
    process.load('DQMServices.Core.DQMStore_cfi')
    process.load('DQM.HGCAL.hgcalDQM_cff')
    process.hgcalrechitdqm.RecHits = 'hgcalSoARecHits'
    process.hgcallayerclusterdqm.RecHits = 'hgcalSoARecHits'
    process.hgcalTestBeamDQMPath = cms.Path(process.hgcalDQMSources + process.hgcalRecoDQMSources)
    process.schedule.insert(1, process.hgcalTestBeamDQMPath)
    process.DQMoutput = cms.OutputModule('DQMRootOutputModule',
                                         fileName=cms.untracked.string('step1_inDQM.root'))
    process.DQMoutput_step = cms.EndPath(process.DQMoutput)
    process.schedule.append(process.DQMoutput_step)

    # Keep HGCal output
    process.FEVTDEBUGoutput.outputCommands = ['keep *_hgcal*_*_*']

    return process


# 2025 and 2026 test beams: digis read from RAW2DIGI/RAW2RECO files (no unpacker), rechits with the CM DNN
# (energy_variations CM_ALT) and DQM with the CM_ALT histograms.
# Maps and calibrations are in cms-data RecoLocalCalo-HGCalRecProducers, see the README in each folder.
_tbData = 'RecoLocalCalo/HGCalRecProducers/data/'
_tbConditions = {
    2025: dict(modules=_tbData + '2025TB/modulelocator_v6.txt',
               calib=_tbData + '2025TB/level0_calib_params_v5.json',
               energyloss=_tbData + '2025TB/hgcal_energyloss_setup2_v1.json',
               modconfig=_tbData + '2025TB/config_econds_v1.json'),
    2026: dict(modules=_tbData + '2026TB/modulelocator_v2.txt',
               sicells=_tbData + '2026TB/WaferCellMapTraces.txt',
               sipmcells=_tbData + '2026TB/channels_sipmontile_tb.hgcal.txt',
               calib=_tbData + '2026TB/level0_calib_params_relay117136.json',
               energyloss=_tbData + '2026TB/hgcal_energyloss_trivial.json',
               modconfig=_tbData + '2026TB/config_econds_v1.json'),
}


def setupConditionsForTB(process, year : int):

    conditions = _tbConditions[year]
    process.load('Configuration.StandardSequences.Accelerators_cff')
    process.load('Configuration.Geometry.GeometryExtendedRun4D122Reco_cff')
    process.load('Configuration.Geometry.GeometryExtendedRun4D122_cff')
    from Geometry.HGCalMapping.hgcalmapping_cff import customise_hgcalmapper
    process = customise_hgcalmapper(
        process, **{k: conditions[k] for k in ('modules', 'sicells', 'sipmcells') if k in conditions})

    # FED and ECON-D configuration, used by the DQM harvester (the FED one is the same for all test beams)
    process.hgcalConfigESProducer = cms.ESSource(
        "HGCalConfigurationESProducer", bePassthroughMode=cms.int32(-1),
        cbHeaderMarker=cms.int32(-1),
        charMode=cms.int32(-1),
        econdHeaderMarker=cms.int32(-1),
        fedjson=cms.FileInPath(_tbData + 'testbeam/config_feds_v1.json'),
        indexSource=cms.ESInputTag("hgCalMappingESProducer", ""),
        modjson=cms.FileInPath(conditions['modconfig']),
        slinkHeaderMarker=cms.int32(-1))

    # ESProducer to load calibration parameters from JSON file, like pedestals
    process.hgcalCalibParamESProducer = cms.ESProducer(
        'hgcalrechit::HGCalCalibrationESProducer@alpaka',
        filename=cms.FileInPath(conditions['calib']),
        filenameEnergyLoss=cms.FileInPath(conditions['energyloss']),
        indexSource=cms.ESInputTag('hgCalMappingESProducer', ''),
        mapSource=cms.ESInputTag('hgCalMappingModuleESProducer', ''))

    # per-module parameters and z-score constants of the CM DNN
    process.hgcalCMMLParamESProducer = cms.ESProducer('HGCalCMMLParamESProducer@alpaka')

    return process


def setupConditionsForTB2025(process):
    return setupConditionsForTB(process, 2025)


def setupConditionsForTB2026(process):
    return setupConditionsForTB(process, 2026)


def runRecoForTB(process, year : int, withLC : bool):

    process = setupConditionsForTB(process, year)

    # Digis come from the input file. Drop earlier RECO/NANO products and the test-system (trigger time, MCP)
    # products, whose classes are defined outside CMSSW.
    process.source.inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_hgcalTrigTimeProducer_*_*',
        'drop *_hgcalRecHits_*_*',
        'drop nanoaod*_*_*_*',
        'drop edmTriggerResults_*_*_*')

    # k_noise=-100 keeps every channel: the CM correction acts on the noise
    process.hgcalSoARecHits = cms.EDProducer('HGCalRecHitsProducer@alpaka',
                                             digis=cms.InputTag('hgcalDigis', ''),
                                             calibSource=cms.ESInputTag('hgcalCalibParamESProducer', ''),
                                             n_hits_scale=cms.int32(1),
                                             n_blocks=cms.int32(1024),
                                             n_threads=cms.int32(1024),
                                             k_noise=cms.double(-100.),
                                             useCMML=cms.bool(True)
                                             )
    process.reco_task = cms.Task(process.hgcalSoARecHits)

    if withLC:
        from RecoTICL.LayerClustering.hgCalSoARecHitsLayerClustersProducer_cfi import hgCalSoARecHitsLayerClustersProducer
        process.hgcalSoARecHitsLayerClusters = hgCalSoARecHitsLayerClustersProducer.clone(
            hgcalRecHitsSoA="hgcalSoARecHits"
        )

        from RecoTICL.LayerClustering.hgCalSoALayerClustersProducer_cfi import hgCalSoALayerClustersProducer
        process.hgcalSoALayerClusters = hgCalSoALayerClustersProducer.clone(
            hgcalRecHitsLayerClustersSoA="hgcalSoARecHitsLayerClusters",
            hgcalRecHitsSoA="hgcalSoARecHits"
        )

        from RecoTICL.LayerClustering.hgCalLayerClustersFromSoAProducer_cfi import hgCalLayerClustersFromSoAProducer
        process.hgcalMergeLayerClusters = hgCalLayerClustersFromSoAProducer.clone(
            hgcalRecHitsLayerClustersSoA="hgcalSoARecHitsLayerClusters",
            hgcalRecHitsSoA="hgcalSoARecHits",
            src="hgcalSoALayerClusters"
        )
        process.reco_task.add(process.hgcalSoARecHitsLayerClusters,
                              process.hgcalSoALayerClusters,
                              process.hgcalMergeLayerClusters)

    process.hgcalTestBeamLocalRecoSequence = cms.Path(process.reco_task)
    process.schedule.insert(0, process.hgcalTestBeamLocalRecoSequence)

    # HGCal DQM clients, saved as DQMIO. No hgcalfaststreamdqm: these files hold no FEDRawDataCollection.
    process.load('DQMServices.Core.DQMStore_cfi')
    process.load('DQM.HGCAL.hgcalDQM_cff')
    process.hgcalrechitdqm.RecHits = 'hgcalSoARecHits'
    process.hgcalrechitdqm.fillCMAlt = cms.bool(True)
    dqm = process.hgcaldigidqm + process.hgcalrechitdqm
    if withLC:
        process.hgcallayerclusterdqm.RecHits = 'hgcalSoARecHits'
        dqm += process.hgcallayerclusterdqm
    process.hgcalTestBeamDQMPath = cms.Path(dqm)
    process.schedule.insert(1, process.hgcalTestBeamDQMPath)
    process.DQMoutput = cms.OutputModule('DQMRootOutputModule',
                                         fileName=cms.untracked.string('step1_inDQM.root'))
    process.DQMoutput_step = cms.EndPath(process.DQMoutput)
    process.schedule.append(process.DQMoutput_step)

    # Keep HGCal output
    process.FEVTDEBUGoutput.outputCommands = ['keep *_hgcal*_*_*']

    return process


def runRecoForTB2025(process):
    return runRecoForTB(process, 2025, withLC=True)


def runRecoForTB2026(process):
    # no layer clusters: the tileboard hits (eta/phi) all fall in one CLUE tile and overflow it
    return runRecoForTB(process, 2026, withLC=False)
