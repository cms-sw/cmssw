import FWCore.ParameterSet.Config as cms

from RecoMET.METProducers.genMht_cff import genMht

genMHTTask = cms.Task(genMht)
genMHT = cms.Sequence(genMHTTask)
