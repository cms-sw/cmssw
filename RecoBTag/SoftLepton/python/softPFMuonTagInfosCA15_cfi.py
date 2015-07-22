import FWCore.ParameterSet.Config as cms

from RecoBTag.SoftLepton.softPFMuonTagInfos_cfi import *

softPFMuonsTagInfosCA15 = softPFMuonsTagInfos.clone(
    jets = cms.InputTag("ca15PFJetsCHS")
)
