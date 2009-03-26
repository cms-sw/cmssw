import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOffline_SecondStep_cff import *
from DQMOffline.Configuration.DQMOffline_Certification_cff import *

from Validation.Configuration.postValidation_cff import *
from HLTriggerOffline.Common.HLTValidationHarvest_cff import *

dqmHarvesting = cms.Path(DQMOffline_SecondStep*DQMOffline_Certification)

validationHarvesting = cms.Path(postValidation*hltpostvalidation)

validationHarvestingPU = cms.Path(postValidation_pu)

#alcaHarvesting = cms.Path()
