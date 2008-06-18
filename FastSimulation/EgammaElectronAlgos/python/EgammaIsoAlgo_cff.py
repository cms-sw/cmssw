import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egammaCalExtractorBlocks_cff import *

EgammaIsoEcalFromHitsExtractorBlock.barrelRecHits = 'caloRecHits:EcalRecHitsEB'
EgammaIsoEcalFromHitsExtractorBlock.endcapRecHits = 'caloRecHits:EcalRecHitsEE'
EgammaIsoHcalFromHitsExtractorBlock.hcalRecHits = 'caloRecHits'

from RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff import *
