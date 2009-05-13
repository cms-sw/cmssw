
from HLTriggerOffline.Common.HLTValidationHarvest_cff import *

from Validation.Configuration.postValidation_cff import *

HarvestingFastSim = cms.Sequence(postvValidation_fastsim
                                 + hltpostvalidation_fastsim)

