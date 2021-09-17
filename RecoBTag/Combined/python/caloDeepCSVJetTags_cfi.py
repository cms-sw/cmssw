import FWCore.ParameterSet.Config as cms
from RecoBTag.Combined.pfDeepCSVJetTags_cfi import pfDeepCSVJetTags

caloDeepCSVJetTags = pfDeepCSVJetTags.clone(src = 'caloDeepCSVTagInfos')
