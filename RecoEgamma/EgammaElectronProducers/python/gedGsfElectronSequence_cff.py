import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to the GED electrons.
#==============================================================================

from RecoEgamma.EgammaElectronProducers.gedGsfElectronCores_cfi import *
from RecoEgamma.EgammaElectronProducers.gedGsfElectrons_cfi import *

gedGsfElectronSequenceTmp = cms.Sequence(gedGsfElectronCores*gedGsfElectronsTmp)

