import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
# $Id: photonSequence_cff.py,v 1.2 2008/04/21 03:26:39 rpw Exp $
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.photonCore_cfi import *
from RecoEgamma.EgammaPhotonProducers.photons_cfi import *
photonSequence = cms.Sequence(photonCore+photons)

