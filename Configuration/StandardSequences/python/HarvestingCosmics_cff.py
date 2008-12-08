import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOfflineCosmics_SecondStep_cff import *
from DQMOffline.Configuration.DQMOfflineCosmics_Certification_cff import *

from Validation.Configuration.postValidation_cff import *

dqmHarvesting = cms.Path(DQMOfflineCosmics_SecondStep*DQMOfflineCosmics_Certification)

validationHarvesting = cms.Path(postValidation)

#alcaHarvesting = cms.Path()
