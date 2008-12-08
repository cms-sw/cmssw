import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOffline_SecondStep_cff import *
from DQMOffline.Configuration.DQMOffline_Certification_cff import *

from Validation.Configuration.postValidation_cff import *

dqmHarvesting = cms.Path(DQMOffline_SecondStep*DQMOffline_Certification)

validationHarvesting = cms.Path(postValidation)

#alcaHarvesting = cms.Path()
