import FWCore.ParameterSet.Config as cms
siPixelHeterogeneousDQMHarvesting = cms.Sequence() # empty sequence if not both CPU and GPU recos are run

from DQM.SiPixelPhase1Common.SiPixelPhase1RawData_cfi import *
from DQM.SiPixelHeterogeneous.SiPixelHeterogenousDQM_FirstStep_cff import SiPixelPhase1RawDataConfForCPU,SiPixelPhase1RawDataConfForGPU,SiPixelPhase1RawDataConfForSerial,SiPixelPhase1RawDataConfForDevice

# CUDA code
siPixelPhase1RawDataHarvesterCPU = SiPixelPhase1RawDataHarvester.clone(histograms = SiPixelPhase1RawDataConfForCPU)
siPixelPhase1RawDataHarvesterGPU = SiPixelPhase1RawDataHarvester.clone(histograms = SiPixelPhase1RawDataConfForGPU)

# alpaka code
siPixelPhase1RawDataHarvesterSerial = SiPixelPhase1RawDataHarvester.clone(histograms = SiPixelPhase1RawDataConfForSerial)
siPixelPhase1RawDataHarvesterDevice = SiPixelPhase1RawDataHarvester.clone(histograms = SiPixelPhase1RawDataConfForDevice)

from DQM.SiPixelHeterogeneous.siPixelTrackComparisonHarvester_cfi import *
siPixelTrackComparisonHarvesterAlpaka = siPixelTrackComparisonHarvester.clone(topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCompareDeviceVSHost'))

siPixelHeterogeneousDQMComparisonHarvesting = cms.Sequence(siPixelPhase1RawDataHarvesterCPU *
                                                           siPixelPhase1RawDataHarvesterGPU *
                                                           siPixelTrackComparisonHarvester )

siPixelHeterogeneousDQMComparisonHarvestingAlpaka = cms.Sequence(siPixelPhase1RawDataHarvesterSerial *
                                                                 siPixelPhase1RawDataHarvesterDevice *
                                                                 siPixelTrackComparisonHarvesterAlpaka )

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
_siPixelHeterogeneousDQMComparisonHarvestingAlpakaPhase2 = cms.Sequence(siPixelTrackComparisonHarvesterAlpaka )

phase2_tracker.toReplaceWith(siPixelHeterogeneousDQMComparisonHarvestingAlpaka,_siPixelHeterogeneousDQMComparisonHarvestingAlpakaPhase2)

# add the harvester in case of the validation modifier is active
from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(siPixelHeterogeneousDQMHarvesting,siPixelHeterogeneousDQMComparisonHarvesting)

from Configuration.ProcessModifiers.alpakaValidationPixel_cff import alpakaValidationPixel
(alpakaValidationPixel & ~gpuValidationPixel).toReplaceWith(siPixelHeterogeneousDQMHarvesting,siPixelHeterogeneousDQMComparisonHarvestingAlpaka)
