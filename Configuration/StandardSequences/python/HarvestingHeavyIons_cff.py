import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOfflineHeavyIons_SecondStep_cff import *
from DQMOffline.Configuration.DQMOfflineHeavyIons_Certification_cff import *

from Validation.RecoHI.HarvestingHI_cff import *

dqmHarvesting = cms.Path(DQMOfflineHeavyIons_SecondStep*DQMOfflineHeavyIons_Certification)
dqmHarvestingPOG = cms.Path(DQMOfflineHeavyIons_SecondStep_PrePOG)
dqmHarvestingFakeHLT =  cms.Path(DQMOfflineHeavyIons_SecondStep_FakeHLT*DQMOfflineHeavyIons_Certification)

validationHarvesting = cms.Path(postValidationHI)
validationHarvestingHI = cms.Path(postValidationHI) ## for backwards compatibility?
