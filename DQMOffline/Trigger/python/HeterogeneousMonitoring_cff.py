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
    topFolderName = cms.string('HLT/HeterogeneousComparisons/PixelErrors'),
    pixelErrorSrcGPU = cms.InputTag("hltSiPixelDigiErrors"),
    pixelErrorSrcCPU = cms.InputTag("hltSiPixelDigiErrorsSerialSync")
)

# Ecal

# TBD

# Hcal
from DQM.HcalTasks.hcalGPUComparisonTask_cfi import *

hltHcalGPUComparisonTask = hcalGPUComparisonTask.clone(
    subsystem = "HLT",
    tagHBHE_ref = "hltHbherecoSerialSync",
    tagHBHE_target = "hltHbhereco"
)

HLT_HeterogeneousMonitoringSequence = cms.Sequence(
    hltPfHcalGPUComparisonTask+
    hltSiPixelPhase1CompareDigiErrors+
    hltHcalGPUComparisonTask    
)
