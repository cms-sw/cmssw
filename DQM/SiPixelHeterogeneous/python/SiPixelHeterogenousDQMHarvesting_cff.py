import FWCore.ParameterSet.Config as cms
siPixelHeterogeneousDQMHarvesting = cms.Sequence() # empty sequence if not both CPU and GPU recos are run

from DQM.SiPixelPhase1Common.SiPixelPhase1RawData_cfi import *
from DQM.SiPixelHeterogeneous.SiPixelHeterogenousDQM_FirstStep_cff import SiPixelPhase1RawDataConfForCPU,SiPixelPhase1RawDataConfForGPU

siPixelPhase1RawDataHarvesterCPU = SiPixelPhase1RawDataHarvester.clone(histograms = SiPixelPhase1RawDataConfForCPU)
siPixelPhase1RawDataHarvesterGPU = SiPixelPhase1RawDataHarvester.clone(histograms = SiPixelPhase1RawDataConfForGPU)

from DQM.SiPixelHeterogeneous.siPixelTrackComparisonHarvester_cfi import *

siPixelHeterogeneousDQMComparisonHarvesting = cms.Sequence(siPixelPhase1RawDataHarvesterCPU *
                                                           siPixelPhase1RawDataHarvesterGPU *
                                                           siPixelTrackComparisonHarvester )

# add the harvester in case of the validation modifier is active
from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(siPixelHeterogeneousDQMHarvesting,siPixelHeterogeneousDQMComparisonHarvesting)


