import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOffline_SecondStep_cff import *
from DQMOffline.Configuration.DQMOffline_Certification_cff import *

from Validation.Configuration.postValidation_cff import *
from HLTriggerOffline.Common.HLTValidationHarvest_cff import *

from FastSimulation.Configuration.Harvesting_cff import *

from Validation.RecoHI.HarvestingHI_cff import *
from Validation.RecoJets.JetPostProcessor_cff import *
from Validation.RecoMET.METPostProcessor_cff import *


dqmHarvesting = cms.Path(DQMOffline_SecondStep*DQMOffline_Certification)
dqmHarvestingPOG = cms.Path(DQMOffline_SecondStep_PrePOG)

dqmHarvestingPOGMC = cms.Path( DQMOffline_SecondStep_PrePOGMC )

validationHarvesting = cms.Path(postValidation*hltpostvalidation*postValidation_gen)

validationpreprodHarvesting = cms.Path(postValidation_preprod*hltpostvalidation_preprod*postValidation_gen)

# empty (non-hlt) postvalidation sequence here yet
validationprodHarvesting = cms.Path(hltpostvalidation_prod*postValidation_gen)

validationHarvestingFS = cms.Path(HarvestingFastSim)

validationHarvestingHI = cms.Path(postValidationHI)

genHarvesting = cms.Path(postValidation_gen)

alcaHarvesting = cms.Path()

validationHarvestingMiniAOD = cms.Path(JetPostProcessor*METPostProcessorHarvesting)
