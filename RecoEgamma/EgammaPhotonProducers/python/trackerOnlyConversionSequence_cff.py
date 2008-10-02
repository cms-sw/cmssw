import FWCore.ParameterSet.Config as cms

#
#
# Tracker only conversion producer
from RecoEgamma.EgammaPhotonProducers.trackerOnlyConversions_cfi import *
trackerOnlyConversionSequence = cms.Sequence(trackerOnlyConversions)


