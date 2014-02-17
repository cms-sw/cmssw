import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
# $Id: photonSequence_CSA06_cff.py,v 1.2 2008/04/21 03:26:40 rpw Exp $
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.photons_cfi import *
# producer for photons after correction
#include "RecoEgamma/EgammaPhotonProducers/data/correctedPhotons.cfi"
# ShR 26 Sep: removed correction to prevent crash in CSA production in 101
photonSequence = cms.Sequence(photons)

