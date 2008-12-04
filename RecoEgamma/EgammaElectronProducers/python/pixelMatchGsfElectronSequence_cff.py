# The following comments couldn't be translated into the new config version:

# FIXME
import FWCore.ParameterSet.Config as cms

# $Id: pixelMatchGsfElectronSequence_cff.py,v 1.3 2008/05/19 23:54:05 rpw Exp $
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
pixelMatchGsfElectronSequence = cms.Sequence(electronPixelSeeds*egammaCkfTrackCandidates*pixelMatchGsfFit*pixelMatchGsfElectrons)

