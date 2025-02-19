import FWCore.ParameterSet.Config as cms

#==============================================================================
# The current official sequence is based on pixels
#==============================================================================

from RecoEgamma.EgammaElectronProducers.gsfElectronSequence_cff import *
# David Chamont:
#   1) original code: electronSequence = cms.Sequence(gsfElectronSequence)
#   2) what I would prefer: electronSequence = gsfElectronSequence
#   3) what we have used for the moment: electronSequence = gsfElectronSequence.copy()
#      see: https://hypernews.cern.ch/HyperNews/CMS/get/recoDevelopment/849.html
electronSequence = gsfElectronSequence.copy()


#==============================================================================
# An alternative sequence based on si-strip
#==============================================================================

from RecoEgamma.EgammaElectronProducers.siStripElectronSequence_cff import *

