import FWCore.ParameterSet.Config as cms

#
#
# Conversion Track candidate producer 
from RecoEgamma.EgammaPhotonProducers.conversionTracks_cff import *
# converted photon producer
#from RecoEgamma.EgammaTools.PhotonConversionMVAComputer_cfi import *
from RecoEgamma.EgammaPhotonProducers.conversions_cfi import *
#conversionSequence = cms.Sequence(ckfTracksFromConversions*conversions)
conversionSequence = cms.Sequence(conversions)

