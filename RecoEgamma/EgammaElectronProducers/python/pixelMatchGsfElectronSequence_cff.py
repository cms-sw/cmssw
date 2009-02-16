import FWCore.ParameterSet.Config as cms

# $Id: pixelMatchGsfElectronSequence_cff.py,v 1.6.2.1 2009/02/16 00:33:52 chamont Exp $
# create a sequence with all required modules and sources needed to make
# Gsf electron sequence
#
# module to make seeds
from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeedsModules_cff import *
# module to make track candidates
#from RecoEgamma.EgammaElectronProducers.gsfElectronCkfTrackCandidateMaker_cff import *
# module to make gsf tracks (track fit)
#from RecoEgamma.EgammaElectronProducers.gsfElectronGsfFit_cff import *
# module to make electrons
from RecoEgamma.EgammaElectronProducers.pixelMatchGsfElectrons_cff import *
pixelMatchGsfElectronSequence = cms.Sequence(pixelMatchGsfElectrons)

