import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to the GED electrons.
#==============================================================================

gedGsfElectronCores = cms.EDProducer("GEDGsfElectronCoreProducer",
    GEDEMUnbiased = cms.InputTag("particleFlowEGamma"),
    gsfTracks = cms.InputTag("electronGsfTracks"),
    ctfTracks = cms.InputTag("generalTracks"),
)

from RecoEgamma.EgammaElectronProducers.gedGsfElectrons_cfi import *

gedGsfElectronSequenceTmp = cms.Sequence(gedGsfElectronCores*gedGsfElectronsTmp)

