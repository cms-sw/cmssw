import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
# $Id: photonSequence_CSA06.cff,v 1.1 2006/09/27 12:37:44 rahatlou Exp $
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.photons_cfi import *
# producer for photons after correction
#include "RecoEgamma/EgammaPhotonProducers/data/correctedPhotons.cfi"
# ShR 26 Sep: removed correction to prevent crash in CSA production in 101
photonSequence = cms.Sequence(photons)

