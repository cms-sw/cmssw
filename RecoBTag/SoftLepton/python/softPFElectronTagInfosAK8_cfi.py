import FWCore.ParameterSet.Config as cms

from RecoBTag.SoftLepton.softPFElectronTagInfos_cfi import *

softPFElectronsTagInfosAK8 = softPFElectronsTagInfos.clone(
    jets = cms.InputTag("ak8PFJetsCHS"),
    DeltaRElectronJet=cms.double(0.8)
)
