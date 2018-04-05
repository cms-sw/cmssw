import FWCore.ParameterSet.Config as cms

modifiedElectrons = cms.EDProducer(
    "ModifiedElectronProducer",
    src = cms.InputTag("slimmedElectrons",processName=cms.InputTag.skipCurrentProcess()),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

from RecoEgamma.EgammaTools.egammaObjectModificationsPatches_cff import *
modifiedElectrons.modifierConfig.modifications = egamma_modifications
