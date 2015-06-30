import FWCore.ParameterSet.Config as cms

slimmedPhotons = cms.EDProducer(
    "ModifiedPhotonProducer",
    src = cms.InputTag("slimmedPhotons",processName=cms.InputTag.skipCurrentProcess()),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

from RecoEgamma.EgammaTools.egammaObjectModificationsPAT_cff import *
slimmedPhotons.modifierConfig.modifications = egamma_modifications
