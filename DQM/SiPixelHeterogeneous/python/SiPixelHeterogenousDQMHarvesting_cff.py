import FWCore.ParameterSet.Config as cms
siPixelHeterogeneousDQMHarvesting = cms.Sequence() # empty sequence if not both CPU and GPU recos are run

from DQM.SiPixelHeterogeneous.siPixelTrackComparisonHarvester_cfi import *
siPixelHeterogeneousDQMComparisonHarvesting = cms.Sequence(siPixelTrackComparisonHarvester)

# add the harvester in case of the validation modifier is active
from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(siPixelHeterogeneousDQMHarvesting,siPixelHeterogeneousDQMComparisonHarvesting)


