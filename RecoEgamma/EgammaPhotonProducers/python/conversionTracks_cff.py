import FWCore.ParameterSet.Config as cms

#
#
# Conversion Track candidate producer 
from RecoEgamma.EgammaPhotonProducers.conversionTrackCandidates_cfi import *
# Conversion Track producer  ( final fit )
#include "RecoEgamma/EgammaPhotonProducers/data/ckfOutInTracksFromConversionsBarrel.cfi"
#include "RecoEgamma/EgammaPhotonProducers/data/ckfInOutTracksFromConversionsBarrel.cfi"
#include "RecoEgamma/EgammaPhotonProducers/data/ckfOutInTracksFromConversionsEndcap.cfi"
#include "RecoEgamma/EgammaPhotonProducers/data/ckfInOutTracksFromConversionsEndcap.cfi"
from RecoEgamma.EgammaPhotonProducers.ckfOutInTracksFromConversions_cfi import *
from RecoEgamma.EgammaPhotonProducers.ckfInOutTracksFromConversions_cfi import *
ckfTracksFromConversions = cms.Sequence(conversionTrackCandidates*ckfOutInTracksFromConversions*ckfInOutTracksFromConversions)

