import FWCore.ParameterSet.Config as cms

slimmedElectrons = cms.EDProducer(
    "ModifiedElectronProducer",
    src = cms.InputTag("slimmedElectrons",processName=cms.InputTag.skipCurrentProcess()),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

from RecoEgamma.EgammaTools.egammaObjectModificationsPAT_cff import *
slimmedElectrons.modifierConfig.modifications = egamma_modifications
