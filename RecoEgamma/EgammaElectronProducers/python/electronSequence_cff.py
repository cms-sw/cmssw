import FWCore.ParameterSet.Config as cms

#==============================================================================
# The current official sequence is based on pixels
#==============================================================================

from RecoEgamma.EgammaElectronProducers.gsfElectronSequence_cff import *
#electronSequence = cms.Sequence(gsfElectronSequence)
electronSequence = gsfElectronSequence


#==============================================================================
# An alternative sequence based on si-strip
#==============================================================================

from RecoEgamma.EgammaElectronProducers.siStripElectronSequence_cff import *

