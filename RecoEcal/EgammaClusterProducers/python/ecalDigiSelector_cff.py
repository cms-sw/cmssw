import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.ecalDigiSelector_cfi import *
seldigis = cms.Sequence(selectDigi)
