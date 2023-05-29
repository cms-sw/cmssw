import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerDefault_cfi import SiPixelClusterizerDefault as _SiPixelClusterizerDefault
siPixelClusters = _SiPixelClusterizerDefault.clone()

# phase1 pixel
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelClusters,
  VCaltoElectronGain      = 47,   # L2-4: 47 +- 4.7
  VCaltoElectronGain_L1   = 50,   # L1:   49.6 +- 2.6
  VCaltoElectronOffset    = -60,  # L2-4: -60 +- 130
  VCaltoElectronOffset_L1 = -670, # L1:   -670 +- 220
  ChannelThreshold        = 10,
  SeedThreshold           = 1000,
  ClusterThreshold        = 4000,
  ClusterThreshold_L1     = 2000
)

# Run3, changes in the gain calibration scheme
#from Configuration.Eras.Era_Run3_cff import Run3
#Run3.toModify(siPixelClusters,
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(siPixelClusters,
  VCaltoElectronGain      = 1,  # all gains=1, pedestals=0
  VCaltoElectronGain_L1   = 1,
  VCaltoElectronOffset    = 0,
  VCaltoElectronOffset_L1 = 0,
  ClusterThreshold_L1     = 4000
)

# Need these until phase2 pixel templates are used
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
from SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi import PixelDigitizerAlgorithmCommon
phase2_tracker.toModify(siPixelClusters, # FIXME
  src = 'simSiPixelDigis:Pixel',
  DropDuplicates = False, # do not drop duplicates for phase-2 until the digitizer can handle them consistently
  MissCalibrate = False,
  Phase2Calibration = True,
  Phase2ReadoutMode = PixelDigitizerAlgorithmCommon.Phase2ReadoutMode.value(), # Flag to decide Readout Mode : linear TDR (-1), dual slope with slope parameters (+1,+2,+3,+4 ...) with threshold subtraction
  Phase2DigiBaseline = PixelDigitizerAlgorithmCommon.ThresholdInElectrons_Barrel.value(), #Same for barrel and endcap
  Phase2KinkADC = 8,
  ElectronPerADCGain = PixelDigitizerAlgorithmCommon.ElectronPerAdc.value()
)
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
(premix_stage2 & phase2_tracker).toModify(siPixelClusters,
    src = "mixData:Pixel"
)
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit
(phase2_tracker & pixelNtupletFit).toModify(siPixelClusters, #at the moment the duplicate dropping is not imnplemented in Phase2
    DropDuplicates = False
)
