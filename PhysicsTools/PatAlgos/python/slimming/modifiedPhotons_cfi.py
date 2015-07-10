import FWCore.ParameterSet.Config as cms

modifiedPhotons = cms.EDProducer(
    "ModifiedPhotonProducer",
    src = cms.InputTag("slimmedPhotons",processName=cms.InputTag.skipCurrentProcess()),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

from RecoEgamma.EgammaTools.egammaObjectModificationsPatches_cff import *
modifiedPhotons.modifierConfig.modifications = egamma_modifications
