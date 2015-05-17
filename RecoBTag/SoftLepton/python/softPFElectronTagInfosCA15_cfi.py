import FWCore.ParameterSet.Config as cms

from RecoBTag.SoftLepton.softPFElectronTagInfos_cfi import *

softPFElectronsTagInfosCA15 = softPFElectronsTagInfos.clone(
    jets = cms.InputTag("ca15PFJetsCHS"),
    DeltaRElectronJet=cms.double(1.5)
)
