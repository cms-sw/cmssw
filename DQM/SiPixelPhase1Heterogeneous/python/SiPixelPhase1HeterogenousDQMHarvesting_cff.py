import FWCore.ParameterSet.Config as cms
siPixelPhase1HeterogenousDQMHarvesting = cms.Sequence() # empty sequence if not both CPU and GPU recos are run

from DQM.SiPixelPhase1Heterogeneous.siPixelPhase1TrackComparisonHarvester_cfi import *
siPixelPhase1HeterogenousDQMComparisonHarvesting = cms.Sequence(siPixelPhase1TrackComparisonHarvester)

# add the harvester in case of the validation modifier is active
from Configuration.ProcessModifiers.gpuValidationPixel_cff import gpuValidationPixel
gpuValidationPixel.toReplaceWith(siPixelPhase1HeterogenousDQMHarvesting,siPixelPhase1HeterogenousDQMComparisonHarvesting)


