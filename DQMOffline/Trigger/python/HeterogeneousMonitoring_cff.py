import FWCore.ParameterSet.Config as cms

# list of products available 
# outputCommands = cms.untracked.vstring( 'drop *',
#       'keep *_hltEcalDigisSerialSync_*_*',
#       'keep *_hltEcalDigis_*_*',
#       'keep *_hltEcalUncalibRecHitSerialSync_*_*',
#       'keep *_hltEcalUncalibRecHit_*_*',
#       'keep *_hltHbherecoSerialSync_*_*',
#       'keep *_hltHbhereco_*_*',
#       'keep *_hltParticleFlowClusterHCALSerialSync_*_*',
#       'keep *_hltParticleFlowClusterHCAL_*_*',
#       'keep *_hltSiPixelDigiErrorsSerialSync_*_*',
#       'keep *_hltSiPixelDigiErrors_*_*' )
# )

# Particle Flow 
from DQM.PFTasks.pfHcalGPUComparisonTask_cfi import *  

hltPfHcalGPUComparisonTask = pfHcalGPUComparisonTask.clone(
    subsystem = cms.untracked.string("HLT"),
    name = cms.untracked.string('HeterogeneousComparisons/ParticleFlow'),
    pfClusterToken_ref = cms.untracked.InputTag('hltParticleFlowClusterHCALSerialSync'),
    pfClusterToken_target = cms.untracked.InputTag('hltParticleFlowClusterHCAL'),   
)

# Tracker
from DQM.SiPixelHeterogeneous.SiPixelHeterogenousDQM_FirstStep_cff import *

hltSiPixelPhase1CompareDigiErrors = siPixelPhase1RawDataErrorComparator.clone(
    pixelErrorSrcGPU = 'hltSiPixelDigiErrors',
    pixelErrorSrcCPU = 'hltSiPixelDigiErrorsSerialSync',
    topFolderName = 'HLT/HeterogeneousComparisons/PixelErrors'
)

hltSiPixelPhase1CompareRecHits = siPixelPhase1CompareRecHits.clone(
    pixelHitsReferenceSoA = 'hltSiPixelRecHitsSoASerialSync',
    pixelHitsTargetSoA  = 'hltSiPixelRecHitsSoA',
    topFolderName = 'HLT/HeterogeneousComparisons/PixelRecHits'
)

hltSiPixelPhase1CompareTracks = siPixelPhase1CompareTracks.clone(
    pixelTrackReferenceSoA = 'hltPixelTracksSoASerialSync',
    pixelTrackTargetSoA = 'hltPixelTracksSoA',
    topFolderName = 'HLT/HeterogeneousComparisons/PixelTracks'
)

hltSiPixelCompareVertices = siPixelCompareVertices.clone(
    pixelVertexReferenceSoA = 'hltPixelVerticesSoASerialSync',
    pixelVertexTargetSoA = 'hltPixelVerticesSoA',
    beamSpotSrc = 'hltOnlineBeamSpot',
    topFolderName = 'HLT/HeterogeneousComparisons/PixelVertices'
)

# Ecal

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import *
from DQM.EcalMonitorTasks.ecalGpuTask_cfi import ecalGpuTask as _ecalGpuTask

_hltdir = 'HLT/HeterogeneousComparison/'

hltEcalGpuTask =  _ecalGpuTask.clone(
    params = _ecalGpuTask.params.clone(
        runGpuTask = True,
        enableRecHit = False
    ),
    MEs = _ecalGpuTask.MEs.clone(
        DigiCpu = _ecalGpuTask.MEs.DigiCpu.clone(path = _hltdir + '%(subdet)s/Digis/%(prefix)sGT digi nDigis cpu'),
        DigiCpuAmplitude = _ecalGpuTask.MEs.DigiCpuAmplitude.clone(path = _hltdir + '%(subdet)s/Digis/%(prefix)sGT digi amplitude sample %(sample)s cpu'),
        DigiGpu = _ecalGpuTask.MEs.DigiGpu.clone(path = _hltdir + '%(subdet)s/Digis/%(prefix)sGT digi nDigis gpu'),
        DigiGpuAmplitude = _ecalGpuTask.MEs.DigiGpuAmplitude.clone(path = _hltdir + '%(subdet)s/Digis/%(prefix)sGT digi amplitude sample %(sample)s gpu'),
        DigiGpuCpu = _ecalGpuTask.MEs.DigiGpuCpu.clone(path = _hltdir + '%(subdet)s/Digis/%(prefix)sGT digi nDigis gpu-cpu diff'),
        DigiGpuCpuAmplitude = _ecalGpuTask.MEs.DigiGpuCpuAmplitude.clone(path = _hltdir + '%(subdet)s/Digis/%(prefix)sGT digi amplitude sample %(sample)s gpu-cpu diff'),
        Digi2D = _ecalGpuTask.MEs.Digi2D.clone(path = _hltdir + '%(subdet)s/Digis/%(prefix)sGT digi nDigis gpu-cpu map2D'),
        Digi2DAmplitude = _ecalGpuTask.MEs.Digi2DAmplitude.clone(path = _hltdir + '%(subdet)s/Digis/%(prefix)sGT digi amplitude sample %(sample)s gpu-cpu map2D'),
        UncalibCpu = _ecalGpuTask.MEs.UncalibCpu.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit nHits cpu'),
        UncalibCpuAmp = _ecalGpuTask.MEs.UncalibCpuAmp.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitude cpu'),
        UncalibCpuAmpError = _ecalGpuTask.MEs.UncalibCpuAmpError.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitudeError cpu'),
        UncalibCpuPedestal = _ecalGpuTask.MEs.UncalibCpuPedestal.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit pedestal cpu'),
        UncalibCpuJitter = _ecalGpuTask.MEs.UncalibCpuJitter.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit jitter cpu'),
        UncalibCpuJitterError = _ecalGpuTask.MEs.UncalibCpuJitterError.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit jitterError cpu'),
        UncalibCpuChi2 = _ecalGpuTask.MEs.UncalibCpuChi2.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit chi2 cpu'),
        UncalibCpuOOTAmp = _ecalGpuTask.MEs.UncalibCpuOOTAmp.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit OOT amplitude %(OOTAmp)s cpu'),
        UncalibCpuFlags = _ecalGpuTask.MEs.UncalibCpuFlags.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit flags cpu'),
        UncalibGpu = _ecalGpuTask.MEs.UncalibGpu.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit nHits gpu'),
        UncalibGpuAmp = _ecalGpuTask.MEs.UncalibGpuAmp.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitude gpu'),
        UncalibGpuAmpError = _ecalGpuTask.MEs.UncalibGpuAmpError.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitudeError gpu'),
        UncalibGpuPedestal = _ecalGpuTask.MEs.UncalibGpuPedestal.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit pedestal gpu'),
        UncalibGpuJitter = _ecalGpuTask.MEs.UncalibGpuJitter.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit jitter gpu'),
        UncalibGpuJitterError = _ecalGpuTask.MEs.UncalibGpuJitterError.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit jitterError gpu'),
        UncalibGpuChi2 = _ecalGpuTask.MEs.UncalibGpuChi2.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit chi2 gpu'),
        UncalibGpuOOTAmp = _ecalGpuTask.MEs.UncalibGpuOOTAmp.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit OOT amplitude %(OOTAmp)s gpu'),
        UncalibGpuFlags = _ecalGpuTask.MEs.UncalibGpuFlags.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit flags gpu'),
        UncalibGpuCpu = _ecalGpuTask.MEs.UncalibGpuCpu.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit nHits gpu-cpu diff'),
        UncalibGpuCpuAmp = _ecalGpuTask.MEs.UncalibGpuCpuAmp.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitude gpu-cpu diff'),
        UncalibGpuCpuAmpError = _ecalGpuTask.MEs.UncalibGpuCpuAmpError.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitudeError gpu-cpu diff'),
        UncalibGpuCpuPedestal = _ecalGpuTask.MEs.UncalibGpuCpuPedestal.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit pedestal gpu-cpu diff'),
        UncalibGpuCpuJitter = _ecalGpuTask.MEs.UncalibGpuCpuJitter.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit jitter gpu-cpu diff'),
        UncalibGpuCpuJitterError = _ecalGpuTask.MEs.UncalibGpuCpuJitterError.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit jitterError gpu-cpu diff'),
        UncalibGpuCpuChi2 = _ecalGpuTask.MEs.UncalibGpuCpuChi2.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit chi2 gpu-cpu diff'),
        UncalibGpuCpuOOTAmp = _ecalGpuTask.MEs.UncalibGpuCpuOOTAmp.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit OOT amplitude %(OOTAmp)s gpu-cpu diff'),
        UncalibGpuCpuFlags = _ecalGpuTask.MEs.UncalibGpuCpuFlags.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit flags gpu-cpu diff'),
        Uncalib2D = _ecalGpuTask.MEs.Uncalib2D.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit nHits gpu-cpu map2D'),
        Uncalib2DAmp = _ecalGpuTask.MEs.Uncalib2DAmp.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitude gpu-cpu map2D'),
        Uncalib2DAmpError = _ecalGpuTask.MEs.Uncalib2DAmpError.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitudeError gpu-cpu map2D'),
        Uncalib2DPedestal = _ecalGpuTask.MEs.Uncalib2DPedestal.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit pedestal gpu-cpu map2D'),
        Uncalib2DJitter = _ecalGpuTask.MEs.Uncalib2DJitter.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit jitter gpu-cpu map2D'),
        Uncalib2DJitterError = _ecalGpuTask.MEs.Uncalib2DJitterError.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit jitterError gpu-cpu map2D'),
        Uncalib2DChi2 = _ecalGpuTask.MEs.Uncalib2DChi2.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit chi2 gpu-cpu map2D'),
        Uncalib2DOOTAmp = _ecalGpuTask.MEs.Uncalib2DOOTAmp.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit OOT amplitude %(OOTAmp)s gpu-cpu map2D'),
        Uncalib2DFlags = _ecalGpuTask.MEs.Uncalib2DFlags.clone(path = _hltdir + '%(subdet)s/UncalibRecHits/%(prefix)sGT uncalib rec hit flags gpu-cpu map2D')
    )
)

hltEcalMonitorTask = ecalMonitorTask.clone(
    workers = ['GpuTask'],
    workerParameters = cms.untracked.PSet(GpuTask = hltEcalGpuTask),
    verbosity = 0,
    commonParameters = ecalMonitorTask.commonParameters.clone(
        willConvertToEDM = False,
        onlineMode = True
    ),
    collectionTags = ecalMonitorTask.collectionTags.clone(
        EcalRawData          = cms.untracked.InputTag("hltEcalDigisSerialSync"),
        EBCpuDigi            = cms.untracked.InputTag("hltEcalDigisSerialSync", "ebDigis"),
        EECpuDigi            = cms.untracked.InputTag("hltEcalDigisSerialSync", "eeDigis"),
        EBGpuDigi            = cms.untracked.InputTag("hltEcalDigis", "ebDigis"),
        EEGpuDigi            = cms.untracked.InputTag("hltEcalDigis", "eeDigis"),
        EBCpuUncalibRecHit   = cms.untracked.InputTag("hltEcalUncalibRecHitSerialSync", "EcalUncalibRecHitsEB"),
        EECpuUncalibRecHit   = cms.untracked.InputTag("hltEcalUncalibRecHitSerialSync", "EcalUncalibRecHitsEE"),
        EBGpuUncalibRecHit   = cms.untracked.InputTag("hltEcalUncalibRecHit", "EcalUncalibRecHitsEB"),
        EEGpuUncalibRecHit   = cms.untracked.InputTag("hltEcalUncalibRecHit", "EcalUncalibRecHitsEE")
    )
)

# Hcal
from DQM.HcalTasks.hcalGPUComparisonTask_cfi import *

hltHcalGPUComparisonTask = hcalGPUComparisonTask.clone(
    subsystem = "HLT",
    tagHBHE_ref = "hltHbherecoSerialSync",
    tagHBHE_target = "hltHbhereco"
)

HLTHeterogeneousMonitoringSequence = cms.Sequence(
    hltPfHcalGPUComparisonTask +
    hltSiPixelPhase1CompareDigiErrors +
    hltSiPixelPhase1CompareRecHits +
    hltSiPixelPhase1CompareTracks +
    hltSiPixelCompareVertices +
    hltEcalMonitorTask +
    hltHcalGPUComparisonTask
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(HLTHeterogeneousMonitoringSequence,cms.Sequence())
