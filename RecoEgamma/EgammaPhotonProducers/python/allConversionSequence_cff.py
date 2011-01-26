import FWCore.ParameterSet.Config as cms

#
#
# Tracker only conversion producer
from RecoEgamma.EgammaPhotonProducers.allConversions_cfi import *
allConversionSequence = cms.Sequence(allConversions)


