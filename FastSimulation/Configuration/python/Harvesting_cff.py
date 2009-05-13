
from HLTriggerOffline.Common.HLTValidationHarvest_cff import *

from Validation.Configuration.postValidation_cff import *

HarvestingFastSim = cms.Sequence(postValidation_fastsim
                                 + hltpostvalidation_fastsim)

