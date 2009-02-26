import FWCore.ParameterSet.Config as cms

#==============================================================================
# The current official sequence is based on pixels
#==============================================================================

from RecoEgamma.EgammaElectronProducers.pixelMatchGsfElectronSequence_cff import *
#electronSequence = cms.Sequence(pixelMatchGsfElectronSequence)
electronSequence = pixelMatchGsfElectronSequence


#==============================================================================
# An alternative sequence based on si-strip
#==============================================================================

from RecoEgamma.EgammaElectronProducers.siStripElectronSequence_cff import *

