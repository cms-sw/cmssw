import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOfflineCosmics_SecondStep_cff import *
from DQMOffline.Configuration.DQMOfflineCosmics_Certification_cff import *

from Validation.Configuration.postValidation_cff import *
from HLTriggerOffline.Common.HLTValidationHarvest_cff import *

dqmHarvesting = cms.Path(DQMOfflineCosmics_SecondStep*DQMOfflineCosmics_Certification)
DQMOfflineCosmics_SecondStep_PrePOG.remove(fsqClient)
dqmHarvestingPOG = cms.Path(DQMOfflineCosmics_SecondStep_PrePOG)

validationHarvesting = cms.Path(postValidation*hltpostvalidation)

#alcaHarvesting = cms.Path()
