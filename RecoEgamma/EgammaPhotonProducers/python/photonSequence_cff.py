import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
# $Id: photonSequence.cff,v 1.9 2008/02/21 17:59:17 nancy Exp $
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.photons_cfi import *
photonSequence = cms.Sequence(photons)

