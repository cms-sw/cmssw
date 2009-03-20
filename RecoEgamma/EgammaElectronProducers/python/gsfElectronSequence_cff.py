import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to make final electrons.
# In the past, this was including the seeding, but this one is directly
# imported in the reco sequences since the integration with pflow.
#==============================================================================

from RecoEgamma.EgammaElectronProducers.gsfElectronModules_cff import *
gsfElectronSequence = cms.Sequence(gsfElectronCores*gsfElectrons)


#==============================================================================
# OBSOLETE
#==============================================================================

# module to make seeds
#from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeedsModules_cff import *
# module to make track candidates
#from RecoEgamma.EgammaElectronProducers.gsfElectronCkfTrackCandidateMaker_cff import *
# module to make gsf tracks (track fit)
#from RecoEgamma.EgammaElectronProducers.gsfElectronGsfFit_cff import *


