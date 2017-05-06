import FWCore.ParameterSet.Config as cms

from RecoBTag.Combined.caloDeepCSVTagInfos_cfi import caloDeepCSVTagInfos
from RecoBTag.Combined.caloDeepCSVJetTags_cfi import caloDeepCSVJetTags

##
## Deep Flavour sequence, not complete as it would need the IP and SV tag infos
##
caloDeepFlavour = cms.Sequence(
    caloDeepCSVTagInfos
    * caloDeepCSVJetTags
)