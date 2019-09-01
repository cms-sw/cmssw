import FWCore.ParameterSet.Config as cms
#
#
# Tracker only conversion producer
from RecoEgamma.EgammaPhotonProducers.allConversions_cfi import *

gsfTracksOpenConversions = allConversions.clone()
gsfTracksOpenConversions.src =  cms.InputTag("gsfTracksOpenConversionTrackProducer")
gsfTracksOpenConversions.AlgorithmName =  cms.string('trackerOnly')
gsfTracksOpenConversions.rCut = cms.double(1.5)
gsfTracksOpenConversions.convertedPhotonCollection = cms.string('gsfTracksOpenConversions')
gsfTracksOpenConversionSequence = cms.Sequence(gsfTracksOpenConversions)
