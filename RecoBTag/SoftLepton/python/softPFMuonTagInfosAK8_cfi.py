import FWCore.ParameterSet.Config as cms

from RecoBTag.SoftLepton.softPFMuonTagInfos_cfi import *

softPFMuonsTagInfosAK8 = softPFMuonsTagInfos.clone(
    jets = cms.InputTag("ak8PFJetsCHS")
)
