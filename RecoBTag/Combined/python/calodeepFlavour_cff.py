import FWCore.ParameterSet.Config as cms

from RecoBTag.Combined.DeepCSVTagInfos_cfi import DeepCSVTagInfos
from RecoBTag.Combined.DeepCSVJetTags_cfi import DeepCSVJetTags

##
## Deep Flavour sequence, not complete as it would need the IP and SV tag infos
##
caloDeepFlavourTask = cms.Task(
    DeepCSVTagInfos,
    ## pfDeepCMVATagInfos, #SKIP for the moment
    DeepCSVJetTags
    ## , pfDeepCMVAJetTags
)
caloDeepFlavour = cms.Sequence(caloDeepFlavourTask)
