import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.ecalDigiSelector_cfi import *
seldigisTask = cms.Task(selectDigi)
seldigis = cms.Sequence(seldigisTask)
