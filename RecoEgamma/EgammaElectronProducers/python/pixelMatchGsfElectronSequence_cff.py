# The following comments couldn't be translated into the new config version:

# FIXME
import FWCore.ParameterSet.Config as cms

# $Id: pixelMatchGsfElectronSequence.cff,v 1.15 2008/04/08 16:38:53 uberthon Exp $
# create a sequence with all required modules and sources needed to make
# Gsf electron sequence
#
# module to make seeds
from RecoEgamma.EgammaElectronProducers.electronPixelSeeds_cff import *
# module to make track candidates
from RecoEgamma.EgammaElectronProducers.gsfElectronCkfTrackCandidateMaker_cff import *
# module to make gsf tracks (track fit)
from RecoEgamma.EgammaElectronProducers.gsfElectronGsfFit_cff import *
# module to make electrons
from RecoEgamma.EgammaElectronProducers.pixelMatchGsfElectrons_cff import *
pixelMatchGsfElectronSequence = cms.Sequence(globalMixedSeeds*electronPixelSeeds*egammaCkfTrackCandidates*pixelMatchGsfFit*pixelMatchGsfElectrons)

