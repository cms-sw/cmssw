import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
# $Id: gedPhotonSequence_cff.py,v 1.1 2013/05/07 12:35:23 nancy Exp $
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.gedPhotonCore_cfi import *
from RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi import *


gedPhotonSequence = cms.Sequence(gedPhotonCore+gedPhotons)

