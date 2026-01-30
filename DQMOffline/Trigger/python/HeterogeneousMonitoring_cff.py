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
_remove = '%(prefix)sGpuTask/'

def cloneMEsWithPathFix(srcMEs, prefix):
    clones = {}
    for name, me in srcMEs.parameters_().items():
        if hasattr(me, 'path'):
            old = me.path.value()
            # remove the unwanted component if present
            new = old.replace(_remove, '')
            # prepend the HLT directory if not already there
            new = prefix + new
            clones[name] = me.clone(path = new)
        else:
            clones[name] = me.clone()
    return clones

hltEcalGpuTask = _ecalGpuTask.clone(
    params = _ecalGpuTask.params.clone(
        runGpuTask = True,
        enableRecHit = False
    ),
    MEs = cloneMEsWithPathFix(_ecalGpuTask.MEs, _hltdir)
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
    #hltSiPixelPhase1CompareRecHits +   # waiting a resolution of #49349
    hltSiPixelPhase1CompareTracks +
    hltSiPixelCompareVertices +
    hltEcalMonitorTask +
    hltHcalGPUComparisonTask    
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(HLTHeterogeneousMonitoringSequence,cms.Sequence())
