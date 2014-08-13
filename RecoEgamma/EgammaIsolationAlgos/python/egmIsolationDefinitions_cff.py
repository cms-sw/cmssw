import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egmGedGsfElectronPFIsolation_cfi import *

egmIsolationSequence = cms.Sequence( egmGedGsfElectronPFIsolation )
