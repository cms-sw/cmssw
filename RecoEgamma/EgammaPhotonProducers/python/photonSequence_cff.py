import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
# $Id: photonSequence_cff.py,v 1.3 2009/03/25 11:15:47 nancy Exp $
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.photonCore_cfi import *
from RecoEgamma.EgammaPhotonProducers.photons_cfi import *
photonSequence = cms.Sequence(photonCore+photons)

