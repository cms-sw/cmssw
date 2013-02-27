import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to make final electrons.
# In the past, this was including the seeding, but this one is directly
# imported in the reco sequences since the integration with pflow.
#==============================================================================

from RecoEgamma.EgammaElectronProducers.gsfElectronModules_cff import *
gsfElectronSequence = cms.Sequence(ecalDrivenGsfElectronCores*ecalDrivenGsfElectrons*gsfElectronCores*gsfElectrons)
gsfEcalDrivenElectronSequence = cms.Sequence(ecalDrivenGsfElectronCores*ecalDrivenGsfElectrons)


#gsfElectronMergingSequence = cms.Sequence(gsfElectronCores*gsfElectrons)

from RecoEgamma.EgammaElectronProducers.edBasedElectronIso_cff import *
from RecoEgamma.EgammaElectronProducers.pfBasedElectronIso_cff import *

electronIsoSequence = cms.Sequence(
        edBasedElectronIsoSequence+
        pfBasedElectronIsoSequence
     )

gsfElectronMergingSequence = cms.Sequence(electronIsoSequence*gsfElectronCores*gsfElectrons)
