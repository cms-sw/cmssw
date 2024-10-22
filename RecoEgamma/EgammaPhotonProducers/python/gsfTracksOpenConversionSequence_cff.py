import FWCore.ParameterSet.Config as cms
#
#
# Tracker only conversion producer
from RecoEgamma.EgammaPhotonProducers.allConversions_cfi import *

gsfTracksOpenConversions = allConversions.clone(
    src           = "gsfTracksOpenConversionTrackProducer",
    AlgorithmName = 'trackerOnly',
    rCut          = 1.5,
    convertedPhotonCollection = 'gsfTracksOpenConversions'
)
gsfTracksOpenConversionSequence = cms.Sequence(gsfTracksOpenConversions)
