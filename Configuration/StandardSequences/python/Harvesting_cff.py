import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOffline_SecondStep_cff import *
from DQMOffline.Configuration.DQMOffline_Certification_cff import *

from Validation.Configuration.postValidation_cff import *
from HLTriggerOffline.Common.HLTValidationHarvest_cff import *

from Validation.RecoHI.HarvestingHI_cff import *
from Validation.RecoJets.JetPostProcessor_cff import *
from Validation.RecoMET.METPostProcessor_cff import *


dqmHarvesting = cms.Path(DQMOffline_SecondStep*DQMOffline_Certification)
dqmHarvestingExtraHLT = cms.Path(DQMOffline_SecondStep_ExtraHLT*DQMOffline_Certification)
dqmHarvestingFakeHLT = cms.Path(DQMOffline_SecondStep_FakeHLT*DQMOffline_Certification)
#dqmHarvesting = cms.Sequence(DQMOffline_SecondStep*DQMOffline_Certification)
#dqmHarvestingFakeHLT = cms.Sequence(DQMOffline_SecondStep_FakeHLT*DQMOffline_Certification)

#dqmHarvestingPOG = cms.Path(DQMOffline_SecondStep_PrePOG)
dqmHarvestingPOG = cms.Sequence(DQMOffline_SecondStep_PrePOG)

dqmHarvestingPOGMC = cms.Path( DQMOffline_SecondStep_PrePOGMC )
#dqmHarvestingPOGMC = cms.Sequence( DQMOffline_SecondStep_PrePOGMC )

validationHarvestingNoHLT = cms.Path(postValidation*postValidation_gen)
validationHarvesting = cms.Path(postValidation*hltpostvalidation*postValidation_gen)
#validationHarvestingNoHLT = cms.Sequence(postValidation*postValidation_gen)
#validationHarvesting = cms.Sequence(postValidation*hltpostvalidation*postValidation_gen)

_validationHarvesting_fastsim = validationHarvesting.copy()
for _entry in [hltpostvalidation]:
    _validationHarvesting_fastsim.remove(_entry)
_validationHarvesting_fastsim.remove(hltpostvalidation)
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(validationHarvesting,_validationHarvesting_fastsim)

validationpreprodHarvestingNoHLT = cms.Path(postValidation_preprod*postValidation_gen)
validationpreprodHarvesting = cms.Path(postValidation_preprod*hltpostvalidation_preprod*postValidation_gen)
#validationpreprodHarvestingNoHLT = cms.Sequence(postValidation_preprod*postValidation_gen)
#validationpreprodHarvesting = cms.Sequence(postValidation_preprod*hltpostvalidation_preprod*postValidation_gen)

_validationpreprodHarvesting_fastsim = validationpreprodHarvesting.copy()
for _entry in [hltpostvalidation_preprod]:
    _validationpreprodHarvesting_fastsim.remove(_entry)
_validationpreprodHarvesting_fastsim.remove(_validationpreprodHarvesting_fastsim)
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(validationpreprodHarvesting,_validationpreprodHarvesting_fastsim)


# empty (non-hlt) postvalidation sequence here yet
validationprodHarvesting = cms.Path(hltpostvalidation_prod*postValidation_gen)
#validationprodHarvesting = cms.Sequence(hltpostvalidation_prod*postValidation_gen)

# to be removed in subsequent request
# kept to avoid too many extra github signatures
validationHarvestingFS = validationHarvestingNoHLT.copy()
validationHarvestingFS.remove(runTauEff) #requires miniAOD Validation

validationHarvestingHI = cms.Path(postValidationHI)
#validationHarvestingHI = cms.Sequence(postValidationHI)

genHarvesting = cms.Path(postValidation_gen)
#genHarvesting = cms.Sequence(postValidation_gen)

alcaHarvesting = cms.Path()
#alcaHarvesting = cms.Sequence()

validationHarvestingMiniAOD = cms.Path(JetPostProcessor*METPostProcessorHarvesting*postValidationMiniAOD)
#validationHarvestingMiniAOD = cms.Sequence(JetPostProcessor*METPostProcessorHarvesting*postValidationMiniAOD)
