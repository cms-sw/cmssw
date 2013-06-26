
from HLTriggerOffline.Common.HLTValidationHarvest_cff import *

from Validation.Configuration.postValidation_cff import *

HarvestingFastSim = cms.Sequence(postValidation_fastsim
                                 + hltpostvalidation_fastsim)

HarvestingFastSim_preprod = cms.Sequence(postValidation_fastsim
                                 + hltpostvalidation_preprod)

HarvestingFastSim_prod = cms.Sequence(hltpostvalidation_prod)


