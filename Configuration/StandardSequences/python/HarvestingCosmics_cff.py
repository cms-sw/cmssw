import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOfflineCosmics_SecondStep_cff import *
from DQMOffline.Configuration.DQMOfflineCosmics_Certification_cff import *

from Validation.Configuration.postValidation_cff import *
from HLTriggerOffline.Common.HLTValidationHarvest_cff import *

dqmHarvesting = cms.Path(DQMOfflineCosmics_SecondStep*DQMOfflineCosmics_Certification)
dqmHarvestingFakeHLT = cms.Path(DQMOfflineCosmics_SecondStep_FakeHLT*DQMOfflineCosmics_CertificationFakeHLT)
dqmHarvestingPOG = cms.Path(DQMOfflineCosmics_SecondStep_PrePOG)

validationHarvesting = cms.Path(postValidationCosmics)

#alcaHarvesting = cms.Path()
